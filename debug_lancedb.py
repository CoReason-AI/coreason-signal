import tempfile

import lancedb

tmp = tempfile.mkdtemp()
db = lancedb.connect(tmp)
db.create_table("test", [{"vector": [1.0, 0.0], "id": "1"}])

tables = db.list_tables()
if hasattr(tables, "tables"):
    print(f"tables.tables: {tables.tables}")
else:
    print("No .tables attribute")
