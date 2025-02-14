
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import random
import os

# Generate Random Level Layouts
def generate_random_levels(size=(10, 10), num_levels=1000):
    print("Generating random levels...")
    levels = []
    for _ in range(num_levels):
        level = np.random.choice([0, 1, 2], size=size)  # 0 = empty, 1 = wall, 2 = enemy
        levels.append(level)
    return np.array(levels)

# Train GAN for Procedural Level Generation
def train_level_generator(levels):
    print("Training GAN for level generation...")

    latent_dim = 100

    # Generator Model
    generator = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(latent_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(np.prod(levels.shape[1:]), activation="sigmoid"),
        layers.Reshape(levels.shape[1:])
    ])

    # Discriminator Model
    discriminator = keras.Sequential([
        layers.Flatten(input_shape=levels.shape[1:]),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    discriminator.trainable = False

    # GAN Model
    gan_input = keras.Input(shape=(latent_dim,))
    generated_level = generator(gan_input)
    gan_output = discriminator(generated_level)

    gan = keras.Model(gan_input, gan_output)
    gan.compile(loss="binary_crossentropy", optimizer="adam")

    # Training Loop
    batch_size = 32
    for epoch in range(10000):
        idx = np.random.randint(0, levels.shape[0], batch_size)
        real_levels = levels[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_levels = generator.predict(noise)

        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_levels, labels_real)
        d_loss_fake = discriminator.train_on_batch(generated_levels, labels_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, labels_real)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss_real[0] + d_loss_fake[0]}, G Loss: {g_loss}")

    generator.save("level_generator.h5")
    print("Level generator model saved.")

# Generate a New Level
def generate_new_level():
    print("Generating new level from trained GAN...")
    generator = keras.models.load_model("level_generator.h5")
    noise = np.random.normal(0, 1, (1, 100))
    generated_level = generator.predict(noise)[0]
    
    plt.imshow(generated_level, cmap="gray")
    plt.title("Generated Game Level")
    plt.savefig("generated_level.png")
    plt.show()

if __name__ == "__main__":
    levels = generate_random_levels()
    train_level_generator(levels)
    generate_new_level()
