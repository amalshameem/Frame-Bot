# Frame Bot V2

## Introduction

A Discord bot that adds a frame to images.

## New features

- A basic editor + preview to see all the frames
- More commands
- Can upload upto 5 frames

## Requirements

- Access to the Discord Developer Portal and a Bot token.
- Python 3.14+.

## Installation

- Create an additional file called config.py and add your bot token there.
- Install all the requirements in requirements.txt (note: all the modules in it might not be needed).
- Run main.py

## Usage

The bot is built with an autoframing mechanism which uses YOLO model to detect subjects. Frames should be of PNG format and transparent on where the image needs to go. They should also be closed frames.

### Usage of AI and Image processing

The program uses AI capabilities to detect the subject and image processing to determine where to put the image. When a frame is uploaded, a mask is created for it. The white part represents where the user-uploaded image will go.

<table style="width: 100%;">
  <tr>
    <td align="center">
      <img src="example/unmasked.png" width="100%">
    </td>
    <td align="center">
      <img src="example/masked.png" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center"><b>Unmasked Image</b></td>
    <td align="center"><b>Masked Image</b></td>
  </tr>
</table>

### Usage in Discord

#### Admin

- Only users with administrator access can setup the bot.
- Add the bot to your server.
- Create a channel for the bot, and set the channel.
- An admin can then upload or remove the frame(upto 5 frames).

#### User

- Can see all the available frames using /frames
- By using the command /useframe, an user can select the frame and upload the image.
- An editor opens up, by default it frames using AI. However if the user want manual controls they can do such here.
- Once finished, the final image will be sent in the channel.

### Commands

- /setchannel (only admin)- Set the channel for framing.
- /uploadframe (only admin)- Upload the transparent PNG frame.
- /frames - Show available frames to choose from.
- /useframe - Upload your image to a specific frame slot.
- /removeframe (only admin)- Delete a frame from a specific slot

# LICENSE

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE)





