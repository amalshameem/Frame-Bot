# Frame Bot

## Introduction

A Discord bot that adds a frame to images.

## Requirements

- Access to the Discord Developer Portal and a Bot token.
- Python 3.14+.

## Installation

- Create an additional file called config.py and add your bot token there.
- Install all the requirements in requirements.txt (note: all the modules in it might not be needed).
- Run main.py

## Usage

The bot is built with an autoframing mechanism which uses YOLO model to detect subjects. For now it mostly works on humans and animals. The program is sensitive to aspect ratio of images. It works best with 1:1 images. Frames should be of PNG format and transparent on where the image needs to go. They should also be closed frames.

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

- Only users with administrator access can interact with the bot's commands.
- Add the bot to your server.
- Create a channel for the bot, and set the channel.
- An admin can then upload the frame.
- When a user sends an image in that channel (one at a time), their image will be automatically framed.

### Commands

- /setchannel - Set the channel for framing.
- /uploadframe - Upload the transparent PNG frame.



