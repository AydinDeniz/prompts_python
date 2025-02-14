
from flask import Flask, request, jsonify
import json
import sqlite3

# Load API configuration from a JSON file
CONFIG_FILE = 'api_config.json'
with open(CONFIG_FILE, 'r') as f:
    api_config = json.load(f)

app = Flask(__name__)

# Initialize SQLite database
DB_FILE = 'database.db'
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create tables based on configuration
for table in api_config['tables']:
    fields = ', '.join([f"{col['name']} {col['type']}" for col in table['columns']])
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table['name']} (id INTEGER PRIMARY KEY AUTOINCREMENT, {fields})")
conn.commit()
conn.close()

# Function to interact with the database
def execute_query(query, params=(), fetch=False):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(query, params)
    if fetch:
        result = cursor.fetchall()
        conn.close()
        return result
    conn.commit()
    conn.close()
    return None

# Dynamically create endpoints based on configuration
for endpoint in api_config['endpoints']:
    route = endpoint['route']
    table = endpoint['table']

    @app.route(route, methods=endpoint['methods'])
    def handle_request(table=table):
        if request.method == 'GET':
            result = execute_query(f"SELECT * FROM {table}", fetch=True)
            return jsonify(result)

        elif request.method == 'POST':
            data = request.get_json()
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?'] * len(data))
            values = tuple(data.values())
            execute_query(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values)
            return jsonify({'message': 'Record added successfully'}), 201

        elif request.method == 'PUT':
            data = request.get_json()
            update_fields = ', '.join([f"{k} = ?" for k in data.keys() if k != 'id'])
            values = tuple(v for k, v in data.items() if k != 'id') + (data['id'],)
            execute_query(f"UPDATE {table} SET {update_fields} WHERE id = ?", values)
            return jsonify({'message': 'Record updated successfully'})

        elif request.method == 'DELETE':
            record_id = request.args.get('id')
            execute_query(f"DELETE FROM {table} WHERE id = ?", (record_id,))
            return jsonify({'message': 'Record deleted successfully'})

if __name__ == '__main__':
    app.run(debug=True)
