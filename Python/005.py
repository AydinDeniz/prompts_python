import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("library.db")
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    author_id INTEGER,
    published_year INTEGER,
    FOREIGN KEY (author_id) REFERENCES authors(id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    membership_date TEXT
)
""")

# Insert sample data into authors table
cursor.execute("INSERT INTO authors (name) VALUES ('J.K. Rowling')")
cursor.execute("INSERT INTO authors (name) VALUES ('George Orwell')")
cursor.execute("INSERT INTO authors (name) VALUES ('J.R.R. Tolkien')")

# Insert sample data into books table
cursor.execute("INSERT INTO books (title, author_id, published_year) VALUES ('Harry Potter and the Philosopher''s Stone', 1, 1997)")
cursor.execute("INSERT INTO books (title, author_id, published_year) VALUES ('Harry Potter and the Chamber of Secrets', 1, 1998)")
cursor.execute("INSERT INTO books (title, author_id, published_year) VALUES ('1984', 2, 1949)")
cursor.execute("INSERT INTO books (title, author_id, published_year) VALUES ('Animal Farm', 2, 1945)")
cursor.execute("INSERT INTO books (title, author_id, published_year) VALUES ('The Hobbit', 3, 1937)")
cursor.execute("INSERT INTO books (title, author_id, published_year) VALUES ('The Lord of the Rings', 3, 1954)")

# Insert sample data into members table
cursor.execute("INSERT INTO members (name, membership_date) VALUES ('Alice', '2023-01-15')")
cursor.execute("INSERT INTO members (name, membership_date) VALUES ('Bob', '2023-03-22')")
cursor.execute("INSERT INTO members (name, membership_date) VALUES ('Charlie', '2023-05-10')")

# Commit the changes
conn.commit()

# Function to retrieve all books by a specific author
def get_books_by_author(author_name):
    cursor.execute("""
    SELECT books.title, books.published_year
    FROM books
    JOIN authors ON books.author_id = authors.id
    WHERE authors.name = ?
    """, (author_name,))
    books = cursor.fetchall()
    if books:
        print(f"Books by {author_name}:")
        for book in books:
            print(f" - {book[0]} ({book[1]})")
    else:
        print(f"No books found for author {author_name}.")

# Example usage of the function
get_books_by_author("J.K. Rowling")

# Close the connection
conn.close()
