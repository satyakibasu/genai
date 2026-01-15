import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('student.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Create a table if it doesn't exist.

if_not_exists_query = "DROP TABLE IF EXISTS STUDENTS;"
cursor.execute(if_not_exists_query)

table_info = """
create table STUDENTS (NAME VARCHAR(255), CLASS VARCHAR(255), SECTION VARCHAR(255), MARKS INT);
"""
cursor.execute(table_info)
print("Table created successfully")

# Insert data into the table
insert_query = """
insert into STUDENTS (NAME, CLASS, SECTION, MARKS) values
('Satyaki', 'Data Science', 'A', 95),
('Ananya', 'DevOps', 'B', 88),
('Rohan', 'Data Science', 'C', 76),
('Priya', 'DevOps', 'A', 89);
"""
cursor.execute(insert_query)
print("Data inserted successfully")

# Display the data
select_query = "SELECT * FROM students;"
cursor.execute(select_query)
rows = cursor.fetchall() #--> this will come in form of list of tuples
print("Data in the table:")
for row in rows:
    print(row)  

# Commit the changes and close the connection
conn.commit()
conn.close()

