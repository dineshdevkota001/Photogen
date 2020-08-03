import PIL.Image as Image

img = Image.open("Bedroom.png").convert('LA').save('newback.png')
