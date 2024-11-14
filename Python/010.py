import random
import string


def generate_secure_key(length):
    # Define the character set: uppercase, lowercase, numbers, and special characters
    characters = string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation

    # Use random.choices to select random characters from the set, ensuring uniqueness
    key = ''.join(random.choices(characters, k=length))
    return key


# Example usage: generate a key of specified length
key_length = 16  # You can specify the length here
secure_key = generate_secure_key(key_length)
print("Generated Secure Key:", secure_key)
