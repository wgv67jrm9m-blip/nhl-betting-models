import duckdb

with duckdb.connect("./nhl_models.duckdb") as con:
	print(con.sql("SHOW TABLES").fetchall())
