import telebot
import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from pydub import AudioSegment
from settings import TOKEN, audio_dir, photo_dir, temp_dir

token = TOKEN
bot = telebot.TeleBot(token)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Send me photo with face or audio and I'll save these files to my database.")


@bot.message_handler(content_types=['audio', 'voice'])
def handle_audio(message):
    """Saves user's audio and voice messages to .wav format sith 16khz frame rate"""

    user_id = message.from_user.id
    file_id = message.audio.file_id if message.content_type == 'audio' else message.voice.file_id
    file_info = bot.get_file(file_id)
    file = bot.download_file(file_info.file_path)
    with open(f"{temp_dir}/audio.ogg", "wb") as f:
        f.write(file)

    audio = AudioSegment.from_file(f"{temp_dir}/audio.ogg", format="ogg")
    audio = audio.set_frame_rate(16000)
    audio.export(get_next_filename(audio_dir, user_id, "audio"), format="wav")
    bot.send_message(message.chat.id, 'Audio saved!')

    os.remove(f"{temp_dir}/audio.ogg")


@bot.message_handler(content_types=['photo', "document"])
def handle_photo(message):
    """Saves user's images with face."""

    if message.content_type == "photo":
        file_id = message.photo[-1].file_id
    else:
        if message.document.mime_type.startswith("image"):
            file_id = message.document.file_id
        else:
            bot.send_message(message.chat.id, "Sorry, it's not an image file.")
            return

    user_id = message.from_user.id
    file_info = bot.get_file(file_id)
    file = bot.download_file(file_info.file_path)

    with open(f"{temp_dir}/image.jpg", "wb") as f:
        f.write(file)

    if classify_face(f"{temp_dir}/image.jpg"):
        with open(get_next_filename(photo_dir, user_id, "photo"), 'wb') as new_file:
            new_file.write(file)
        bot.send_message(message.chat.id, f'Image saved!')
    else:
        bot.send_message(message.chat.id, 'No face found in image.')

    os.remove(f"{temp_dir}/image.jpg")


def get_next_filename(directory: str, user_id: int, file_type: str):
    """Creates filenames for audio and image files depending on which are already exist.
    Returns a filename as [directory]/[type]_message_[user id]_[message number]_[extension]]"""

    extensions = {
        "audio": ".wav",
        "photo": ".jpg"
    }
    i = 1
    while os.path.exists(f"{directory}/{file_type}_message_{user_id}_{i}{extensions[file_type]}"):
        i += 1
    return f"{directory}/{file_type}_message_{user_id}_{i}{extensions[file_type]}"


def classify_face(image_path):
    """Gets an image. Returns does it contain faces."""
    img = Image.open(image_path)
    with torch.no_grad():
        faces = mtcnn(img)
    return faces is not None


if __name__ == "__main__":
    print(f'Running on device: {device}')
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    bot.infinity_polling()
