# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account

import pickle


def crop_image(name):
    # Connect to Cloud Vision API
    credentials = service_account.Credentials.from_service_account_file(
    'C:\\Users\\archi\\Hacktech2020-4c5c4a21d929.json')

    # Instantiates a client
    client = vision.ImageAnnotatorClient(credentials=credentials)

    # Load image
    with open(name + '.jpg', 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations

    polygon = None

    # Gets bounding box of first object
    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        polygon = object_.bounding_poly.normalized_vertices
        for vertex in polygon:
            print(' - ({}, {})'.format(vertex.x, vertex.y))

    # Dumps file for later use
    with open(name + '.p', 'wb') as f:
        pickle.dump(list(polygon), f)
