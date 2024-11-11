import cv2
import numpy as np
import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Face Dodge Game")

# Colors
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Avatar properties
avatar_width, avatar_height = 50, 50
avatar_x = (WIDTH - avatar_width) // 2
avatar_y = HEIGHT - avatar_height - 10

# Falling object properties
falling_objects = []
falling_speed = 5
object_width, object_height = 50, 50

# Load face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to spawn a new falling object
def spawn_object():
    x = random.randint(0, WIDTH - object_width)
    falling_objects.append([x, 0])  # Start at top of the screen

# Game loop
running = True
clock = pygame.time.Clock()

# Initialize video capture with lower resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set to 640x480 for lower resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_detection_interval = 5  # Process face detection every N frames
frame_count = 0

while running:
    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize the frame to fit the Pygame window size
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    # Convert the frame to RGB (Pygame uses RGB, OpenCV uses BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the frame to a surface that Pygame can display
    frame_surface = pygame.surfarray.make_surface(frame_rgb)
    # Rotate the frame to match the Pygame coordinate system
    frame_surface = pygame.transform.rotate(frame_surface, -90)
    # Remove the horizontal flip so the camera feed is not mirrored
    frame_surface = pygame.transform.flip(frame_surface, False, False)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces every few frames
    if frame_count % face_detection_interval == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Move avatar based on face position
        if len(faces) > 0:
            # Get the first detected face
            (x, y, w, h) = faces[0]
            # Center the avatar under the detected face
            avatar_x = x + w // 2 - avatar_width // 2  # Center avatar under the face
            avatar_x = max(0, min(WIDTH - avatar_width, avatar_x))  # Keep within bounds

            # Invert movement direction
            avatar_x = WIDTH - avatar_x - avatar_width

    # Spawn objects at a certain interval
    if random.random() < 0.1:  # Adjust frequency here
        spawn_object()

    # Update falling objects
    for obj in falling_objects:
        obj[1] += falling_speed  # Move down
        if obj[1] > HEIGHT:
            falling_objects.remove(obj)  # Remove off-screen objects

    # Collision detection
    avatar_rect = pygame.Rect(avatar_x, avatar_y, avatar_width, avatar_height)
    for obj in falling_objects:
        object_rect = pygame.Rect(obj[0], obj[1], object_width, object_height)
        if avatar_rect.colliderect(object_rect):
            print("Game Over!")  # For now, just print; you can handle game over logic here
            running = False

    # Drawing
    screen.blit(frame_surface, (0, 0))  # Draw the camera feed as the background
    pygame.draw.rect(screen, GREEN, (avatar_x, avatar_y, avatar_width, avatar_height))  # Draw the avatar
    for obj in falling_objects:
        pygame.draw.rect(screen, RED, (obj[0], obj[1], object_width, object_height))  # Draw falling objects
    
    # Update display
    pygame.display.flip()
    clock.tick(30)  # Limit to 30 frames per second
    frame_count += 1

# Clean up
cap.release()
pygame.quit()
