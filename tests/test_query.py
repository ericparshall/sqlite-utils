import pytest
import types

from sqlite_utils import Database
from sqlite_utils.utils import sqlite3


def test_query(fresh_db):
    fresh_db["dogs"].insert_all([{"name": "Cleo"}, {"name": "Pancakes"}])
    results = fresh_db.query("select * from dogs order by name desc")
    assert isinstance(results, types.GeneratorType)
    assert list(results) == [{"name": "Pancakes"}, {"name": "Cleo"}]


def test_query_executes_eagerly(fresh_db):
    # The SQL runs when query() is called, not when the result is iterated,
    # so errors are raised at the call site
    with pytest.raises(sqlite3.OperationalError):
        fresh_db.query("select * from missing_table")


def test_query_rejects_statements_that_return_no_rows(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    with pytest.raises(ValueError) as ex:
        fresh_db.query("update dogs set name = 'Cleopaws'")
    assert "execute()" in str(ex.value)


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)
def test_query_insert_returning(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    rows = list(
        fresh_db.query("insert into dogs (name) values ('Pancakes') returning name")
    )
    assert rows == [{"name": "Pancakes"}]
    assert fresh_db["dogs"].count == 2


def test_execute_returning_dicts(fresh_db):
    # Like db.query() but returns a list, included for backwards compatibility
    # see https://github.com/simonw/sqlite-utils/issues/290
    fresh_db["test"].insert({"id": 1, "bar": 2}, pk="id")
    assert fresh_db.execute_returning_dicts("select * from test") == [
        {"id": 1, "bar": 2}
    ]


requires_returning = pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)


@pytest.fixture
def file_db(tmp_path):
    path = str(tmp_path / "test.db")
    db = Database(path)
    db["dogs"].insert({"id": 1, "name": "Cleo"}, pk="id")
    yield db, path
    db.close()


def _other_connection_names(path):
    other = sqlite3.connect(path)
    try:
        return [r[0] for r in other.execute("select name from dogs order by id")]
    finally:
        other.close()


def _assert_rejected(db, sql):
    with pytest.raises(ValueError) as ex:
        db.query(sql)
    assert "execute()" in str(ex.value)
    assert not db.conn.in_transaction


@pytest.mark.parametrize(
    "sql",
    (
        "update dogs set name = 'Cleopaws'",
        "insert into dogs (name) values ('Pancakes')",
        "delete from dogs",
        "create table t (id integer primary key)",
        "drop table dogs",
        "create index idx_dogs_name on dogs (name)",
        "-- a comment\nupdate dogs set name = 'Cleopaws'",
    ),
)
def test_query_rejected_statement_has_no_effect(file_db, sql):
    db, path = file_db
    _assert_rejected(db, sql)
    assert db.table_names() == ["dogs"]
    assert db["dogs"].indexes == []
    assert _other_connection_names(path) == ["Cleo"]


@pytest.mark.parametrize(
    "sql",
    (
        "begin",
        "begin immediate",
        "commit",
        "end",
        "rollback",
        "savepoint s",
        "release s",
        "vacuum",
        "attach database ':memory:' as foo",
        "detach database foo",
        "/* comment */ begin",
    ),
)
def test_query_rejects_transaction_control_and_similar(file_db, sql):
    db, _ = file_db
    _assert_rejected(db, sql)
    assert [r[1] for r in db.execute("pragma database_list").fetchall()] == ["main"]


@pytest.mark.parametrize(
    "sql,pragma,expected",
    (
        ("pragma user_version = 5", "user_version", 0),
        ("pragma main.user_version = 5", "user_version", 0),
        ("PRAGMA foreign_keys = on", "foreign_keys", 0),
        ("pragma automatic_index(0)", "automatic_index", 1),
        ("pragma cache_size = 7", "cache_size", -2000),
    ),
)
def test_query_rejects_pragma_without_rows(file_db, sql, pragma, expected):
    db, _ = file_db
    _assert_rejected(db, sql)
    assert db.execute("pragma {}".format(pragma)).fetchone()[0] == expected


def test_query_row_returning_pragmas(file_db):
    db, _ = file_db
    assert list(db.query("pragma journal_mode = wal")) == [{"journal_mode": "wal"}]
    assert db.journal_mode == "wal"
    assert list(db.query("pragma user_version")) == [{"user_version": 0}]
    assert [r["name"] for r in db.query("pragma table_info(dogs)")] == ["id", "name"]
    # Declares columns but returns no rows
    assert list(db.query("pragma index_list(dogs)")) == []
    assert not db.conn.in_transaction


def test_query_rejected_inside_explicit_transaction(file_db):
    db, path = file_db
    db.begin()
    db.execute("insert into dogs (id, name) values (2, 'Pancakes')")
    for sql in (
        "update dogs set name = 'Renamed'",
        "create table t (id integer)",
        "commit",
        "rollback",
        "begin",
    ):
        with pytest.raises(ValueError):
            db.query(sql)
        assert db.conn.in_transaction
    db.commit()
    assert db.table_names() == ["dogs"]
    assert _other_connection_names(path) == ["Cleo", "Pancakes"]


def test_query_error_leaves_no_transaction(file_db):
    db, _ = file_db
    with pytest.raises(sqlite3.OperationalError):
        db.query("select * from missing_table")
    assert not db.conn.in_transaction


@requires_returning
@pytest.mark.parametrize(
    "sql,expected_names",
    (
        (
            "insert into dogs (id, name) values (2, 'A'), (3, 'B') returning name",
            ["Cleo", "A", "B"],
        ),
        ("update dogs set name = 'Cleopaws' returning name", ["Cleopaws"]),
        ("delete from dogs returning name", []),
        (
            "with n as (select 'A' as name) "
            "insert into dogs (id, name) select 2, name from n returning name",
            ["Cleo", "A"],
        ),
    ),
)
@pytest.mark.parametrize("consume", ("none", "partial", "all"))
def test_query_returning_committed_without_iteration(
    file_db, sql, expected_names, consume
):
    db, path = file_db
    rows = db.query(sql)
    assert not db.conn.in_transaction
    assert _other_connection_names(path) == expected_names
    if consume == "partial":
        next(rows, None)
    elif consume == "all":
        returned = list(rows)
        assert all(set(r) == {"name"} for r in returned)
    assert not db.conn.in_transaction
    assert _other_connection_names(path) == expected_names


@requires_returning
def test_query_returning_partially_iterated_yields_rest(file_db):
    db, _ = file_db
    rows = db.query(
        "insert into dogs (id, name) values (2, 'A'), (3, 'B') returning name"
    )
    assert next(rows) == {"name": "A"}
    assert list(rows) == [{"name": "B"}]


@requires_returning
def test_query_returning_inside_explicit_transaction(file_db):
    db, path = file_db
    db.begin()
    rows = list(db.query("insert into dogs (id, name) values (2, 'A') returning id"))
    assert rows == [{"id": 2}]
    assert db.conn.in_transaction
    assert _other_connection_names(path) == ["Cleo"]
    db.rollback()
    assert [r["name"] for r in db["dogs"].rows] == ["Cleo"]


@requires_returning
def test_query_returning_constraint_error_leaves_no_transaction(file_db):
    db, _ = file_db
    with pytest.raises(sqlite3.IntegrityError):
        db.query("insert into dogs (id, name) values (1, 'Dupe') returning id")
    assert not db.conn.in_transaction
