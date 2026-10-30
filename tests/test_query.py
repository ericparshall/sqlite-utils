import pytest
import types

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
    from sqlite_utils import Database

    path = str(tmp_path / "data.db")
    db = Database(path)
    db["dogs"].insert({"name": "Cleo"})
    yield db, path
    db.close()


def _other_connection_names(path):
    conn = sqlite3.connect(path)
    try:
        return [row[0] for row in conn.execute("select name from dogs order by rowid")]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "sql",
    (
        "update dogs set name = 'Cleopaws'",
        "delete from dogs",
        "insert into dogs (name) values ('Pancakes')",
        "create table t (id integer primary key)",
        "drop table dogs",
        "pragma user_version = 5",
        "-- comment\nupdate dogs set name = 'Cleopaws'",
    ),
)
def test_query_rejected_statement_has_no_effect(file_db, sql):
    db, path = file_db
    with pytest.raises(ValueError) as ex:
        db.query(sql)
    assert "execute()" in str(ex.value)
    assert not db.conn.in_transaction
    assert db.table_names() == ["dogs"]
    assert _other_connection_names(path) == ["Cleo"]
    assert db.execute("pragma user_version").fetchone()[0] == 0


@pytest.mark.parametrize(
    "sql",
    (
        "begin",
        "BEGIN IMMEDIATE",
        "commit",
        "end",
        "rollback",
        "savepoint foo",
        "release foo",
        "vacuum",
        "attach database ':memory:' as foo",
        "detach database foo",
        "  /* comment */ begin",
    ),
)
def test_query_rejects_transaction_control(fresh_db, sql):
    with pytest.raises(ValueError) as ex:
        fresh_db.query(sql)
    assert "execute()" in str(ex.value)
    assert not fresh_db.conn.in_transaction
    assert [row[1] for row in fresh_db.execute("pragma database_list")] == ["main"]


def test_query_rejected_statement_inside_transaction(file_db):
    db, path = file_db
    db.begin()
    db["dogs"].insert({"name": "Pancakes"})
    with pytest.raises(ValueError):
        db.query("update dogs set name = 'Cleopaws'")
    assert db.conn.in_transaction
    db.commit()
    assert _other_connection_names(path) == ["Cleo", "Pancakes"]


def test_query_error_leaves_no_transaction(fresh_db):
    with pytest.raises(sqlite3.OperationalError):
        fresh_db.query("select * from missing_table")
    assert not fresh_db.conn.in_transaction


def test_query_select_leaves_no_transaction(fresh_db):
    fresh_db["dogs"].insert_all([{"name": "Cleo"}, {"name": "Pancakes"}])
    rows = fresh_db.query("select * from dogs")
    assert not fresh_db.conn.in_transaction
    assert next(rows) == {"name": "Cleo"}
    assert list(rows) == [{"name": "Pancakes"}]


def test_query_pragma(file_db):
    db, _ = file_db
    assert list(db.query("pragma journal_mode = wal")) == [{"journal_mode": "wal"}]
    assert db.journal_mode == "wal"
    assert list(db.query("pragma table_info(dogs)"))[0]["name"] == "name"
    assert list(db.query("pragma table_info(missing)")) == []
    assert not db.conn.in_transaction
    # A no-rows PRAGMA compiled at prepare time must not take effect
    with pytest.raises(ValueError):
        db.query("pragma foreign_keys = on")
    assert db.execute("pragma foreign_keys").fetchone()[0] == 0


def test_query_pragma_inside_transaction(fresh_db):
    fresh_db.begin()
    assert list(fresh_db.query("pragma user_version")) == [{"user_version": 0}]
    with pytest.raises(ValueError):
        fresh_db.query("pragma user_version = 5")
    fresh_db.commit()
    assert fresh_db.execute("pragma user_version").fetchone()[0] == 0


@requires_returning
@pytest.mark.parametrize("consume", ("none", "partial", "all"))
@pytest.mark.parametrize(
    "sql,expected_rows,expected_names",
    (
        (
            "insert into dogs (name) values ('A'), ('B') returning name",
            [{"name": "A"}, {"name": "B"}],
            ["Cleo", "A", "B"],
        ),
        (
            "update dogs set name = 'Cleopaws' returning name",
            [{"name": "Cleopaws"}],
            ["Cleopaws"],
        ),
        ("delete from dogs returning name", [{"name": "Cleo"}], []),
        ("update dogs set name = 'X' where 0 returning name", [], ["Cleo"]),
    ),
)
def test_query_returning_is_committed(
    file_db, sql, expected_rows, expected_names, consume
):
    db, path = file_db
    rows = db.query(sql)
    assert not db.conn.in_transaction
    assert _other_connection_names(path) == expected_names
    if consume == "partial":
        assert next(rows, None) == (expected_rows[0] if expected_rows else None)
    elif consume == "all":
        assert list(rows) == expected_rows


@requires_returning
def test_query_returning_inside_transaction(file_db):
    db, path = file_db
    db.begin()
    rows = db.query("insert into dogs (name) values ('A'), ('B') returning name")
    assert next(rows) == {"name": "A"}
    assert db.conn.in_transaction
    db.rollback()
    assert _other_connection_names(path) == ["Cleo"]
    assert [r["name"] for r in db["dogs"].rows] == ["Cleo"]


@requires_returning
def test_query_returning_inside_atomic(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    with fresh_db.atomic():
        assert list(
            fresh_db.query("insert into dogs (name) values ('A') returning name")
        ) == [{"name": "A"}]
    assert fresh_db["dogs"].count == 2


def test_query_tracer(fresh_db):
    collected = []
    with fresh_db.tracer(lambda sql, params: collected.append((sql, params))):
        list(fresh_db.query("select ?", [1]))
    assert ("select ?", [1]) in collected
