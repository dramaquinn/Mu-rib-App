#Database
import sqlite3

# create the database
connection = sqlite3.connect("users_data.db")
cursor = connection.cursor()

# table for the users information
command = "CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, " \
          "email TEXT UNIQUE, mobileNum TEXT, password TEXT);"  # Added 'icon BLOB' column


# table to hold the user's tests information
command2 = ("CREATE TABLE IF NOT EXISTS tests (id TEXT PRIMARY KEY , testName TEXT , score INTEGER, username TEXT,"
            "FOREIGN KEY (username) REFERENCES users(username));")


# execute the queries
cursor.execute(command)
cursor.execute(command2)

# Commit the changes and close the connection
connection.commit()
connection.close()



