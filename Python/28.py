
import hashlib
import itertools

# Hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Brute force attack
def brute_force_attack(hash_target, charset, max_length):
    for length in range(1, max_length + 1):
        for attempt in itertools.product(charset, repeat=length):
            attempt_password = ''.join(attempt)
            if hash_password(attempt_password) == hash_target:
                return attempt_password
    return None

# Dictionary attack
def dictionary_attack(hash_target, dictionary_file):
    with open(dictionary_file, "r") as f:
        for line in f:
            word = line.strip()
            if hash_password(word) == hash_target:
                return word
    return None

if __name__ == "__main__":
    target_password = "secret123"  # Replace with actual password
    hash_target = hash_password(target_password)
    
    print("Performing dictionary attack...")
    dictionary_result = dictionary_attack(hash_target, "common_passwords.txt")
    if dictionary_result:
        print(f"Password found: {dictionary_result}")
    else:
        print("Dictionary attack failed.")
    
    print("Performing brute force attack...")
    brute_force_result = brute_force_attack(hash_target, "abcdefghijklmnopqrstuvwxyz1234567890", 6)
    if brute_force_result:
        print(f"Password found: {brute_force_result}")
    else:
        print("Brute force attack failed.")
