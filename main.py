import discord
from discord import app_commands
from discord.ext import commands
import cv2
import numpy as np
from PIL import Image, ImageOps
import io
import aiohttp
import os
import json
from typing import Union
from ultralytics import YOLO

# --- CONFIGURATION ---
try:
    import config
except ImportError:
    print("❌ Error: config.py not found. Please create it and add your TOKEN.")
    exit()

SETTINGS_FILE = "settings.json"
FRAMES_DIR = "frames"

if not os.path.exists(FRAMES_DIR):
    os.makedirs(FRAMES_DIR)

# --- LOAD AI MODEL ---

print("⏳ Loading AI Model... (This might take a moment on first run)")
model = YOLO("yolo11s.pt") 
print("✅ AI Model Loaded.")

# --- PERSISTENCE ---
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)

settings = load_settings()

# --- AI & IMAGE PROCESSING ENGINE ---
class ImageEngine:
    state = {}

    @staticmethod
    def get_paths(guild_id):
        return (
            os.path.join(FRAMES_DIR, f"{guild_id}_frame.png"),
            os.path.join(FRAMES_DIR, f"{guild_id}_mask.png")
        )

    @staticmethod
    def cleanup_old_files(guild_id):
        frame_path, mask_path = ImageEngine.get_paths(guild_id)
        if os.path.exists(frame_path):
            try: os.remove(frame_path)
            except: pass
        if os.path.exists(mask_path):
            try: os.remove(mask_path)
            except: pass
        if guild_id in ImageEngine.state:
            del ImageEngine.state[guild_id]

    @staticmethod
    def analyze_frame_and_save(guild_id):
        frame_path, mask_path = ImageEngine.get_paths(guild_id)
        if not os.path.exists(frame_path): return None

        # 1. Load Frame
        frame_pil = Image.open(frame_path).convert("RGBA")
        img_np = np.array(frame_pil)
        h, w = img_np.shape[:2]
        
        # 2. Extract Alpha
        if img_np.shape[2] == 4:
            alpha = img_np[:, :, 3]
        else:
            alpha = np.where(np.all(img_np < 10, axis=-1), 0, 255).astype(np.uint8)

        # 3. Binary Map
        _, binary = cv2.threshold(alpha, 50, 255, cv2.THRESH_BINARY)
        
        # 4. Find Seed (Hole Logic)
        center_x, center_y = w // 2, h // 2
        seed_point = None

        if binary[center_y, center_x] == 0:
            seed_point = (center_x, center_y)
        else:
            print(f"[{guild_id}] ⚠️ Center blocked. Searching...")
            found = False
            for r in range(10, min(w, h)//4, 10):
                if found: break
                for theta in range(0, 360, 45):
                    x = int(center_x + r * np.cos(np.radians(theta)))
                    y = int(center_y + r * np.sin(np.radians(theta)))
                    if 0 <= x < w and 0 <= y < h:
                        if binary[y, x] == 0:
                            seed_point = (x, y)
                            found = True
                            break
        
        # 5. Flood Fill
        h_mask, w_mask = h + 2, w + 2
        flood_mask = np.zeros((h_mask, w_mask), np.uint8)
        final_mask = np.zeros((h, w), np.uint8)

        if seed_point:
            binary_copy = binary.copy()
            cv2.floodFill(binary_copy, flood_mask, seed_point, 255, flags=4 | (255 << 8))
            final_mask = flood_mask[1:-1, 1:-1]
        else:
            print(f"[{guild_id}] ❌ No hole found. Using full box.")
            final_mask.fill(255)

        # 6. Bbox
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = (0, 0, w, h)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            bbox = (x, y, bw, bh)

        mask_pil = Image.fromarray(final_mask)
        mask_pil.save(mask_path)

        ImageEngine.state[guild_id] = {
            "current_frame": frame_pil,
            "current_mask": mask_pil,
            "frame_hole_bbox": bbox
        }
        return bbox

    @staticmethod
    def detect_subject_center(img_pil):
        """
        Smart Detection:
        1. If Full Body (Tall box) -> Aim at the head (Top 20%).
        2. If Selfie/Headshot (Square box) -> Aim at the nose (Top 45%).
        3. If Animal/Object -> Aim at geometric center.
        """
        w, h = img_pil.size
        img_np = np.array(img_pil.convert('RGB'))
        results = model(img_np, verbose=False)
        
        # 0: person, 15: cat, 16: dog, 14: bird, 87: teddy bear
        target_classes = [0, 15, 16, 14, 87]
        
        boxes = []
        person_boxes = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((x1, y1, x2, y2))
                    if cls == 0:
                        person_boxes.append((x1, y1, x2, y2))
        
        if not boxes:
            return w // 2, h // 2
            
        # Determine which boxes to use for centering
        # Human subjects are prioritized.
        active_boxes = person_boxes if person_boxes else boxes

        min_x = min(b[0] for b in active_boxes)
        min_y = min(b[1] for b in active_boxes)
        max_x = max(b[2] for b in active_boxes)
        max_y = max(b[3] for b in active_boxes)
        
        box_w = max_x - min_x
        box_h = max_y - min_y
        
        center_x = int((min_x + max_x) / 2)

        # --- DYNAMIC HEAD LOGIC ---
        if person_boxes:
            # Calculate Aspect Ratio of the human box
            aspect_ratio = box_w / box_h 
            
            # CASE A: Full Body
            if aspect_ratio < 0.85:
                offset = 0.20 
            
            # CASE B: Selfie 
            else:
                offset = 0.45
            
            center_y = int(min_y + (box_h * offset))
        else:
            # For animals and other objects
            center_y = int((min_y + max_y) / 2)
        
        return center_x, center_y
        
    @staticmethod
    def smart_crop(user_img_pil, target_w, target_h):
        # 1. Find center using YOLO
        cx, cy = ImageEngine.detect_subject_center(user_img_pil)
        
        # 2. Calculate Aspect Ratios
        img_w, img_h = user_img_pil.size
        target_aspect = target_w / target_h
        img_aspect = img_w / img_h

        # 3. Resize Logic (Scale to fill)
        if img_aspect > target_aspect:
            # Image is wider than target -> Height is the constraint
            new_h = target_h
            new_w = int(target_h * img_aspect)
        else:
            # Image is taller than target -> Width is the constraint
            new_w = target_w
            new_h = int(target_w / img_aspect)

        resized_img = user_img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 4. Map the detection center to the new resized coordinates
        scale_x = new_w / img_w
        scale_y = new_h / img_h
        new_cx = int(cx * scale_x)
        new_cy = int(cy * scale_y)

        # 5. Crop based on the detected center
        left = new_cx - (target_w // 2)
        top = new_cy - (target_h // 2)
        
        # 6. Don't crop outside image
        if left < 0: left = 0
        if top < 0: top = 0
        if left + target_w > new_w: left = new_w - target_w
        if top + target_h > new_h: top = new_h - target_h

        return resized_img.crop((left, top, left + target_w, top + target_h))

    @staticmethod
    def process_final_image(user_img_bytes, guild_id):
        if guild_id not in ImageEngine.state:
            frame_path, mask_path = ImageEngine.get_paths(guild_id)
            if os.path.exists(frame_path) and os.path.exists(mask_path):
                 ImageEngine.analyze_frame_and_save(guild_id)
            else:
                return None

        guild_data = ImageEngine.state[guild_id]
        
        user_img = Image.open(io.BytesIO(user_img_bytes)).convert("RGBA")
        frame_img = guild_data["current_frame"]
        mask_img = guild_data["current_mask"] 
        hx, hy, hw, hh = guild_data["frame_hole_bbox"]
        
        cropped_user = ImageEngine.smart_crop(user_img, hw, hh)
        
        content_layer = Image.new("RGBA", frame_img.size, (0, 0, 0, 0))
        content_layer.paste(cropped_user, (hx, hy))
        masked_content = Image.composite(content_layer, Image.new("RGBA", frame_img.size, (0,0,0,0)), mask_img)
        masked_content.alpha_composite(frame_img)
        return masked_content

# --- BOT SETUP ---
class FrameBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="?", intents=intents)

    async def setup_hook(self):
        await self.tree.sync()

bot = FrameBot()

# --- EVENTS (CLEANUP) ---
@bot.event
async def on_guild_remove(guild):
    """Auto-delete data when kicked"""
    guild_id = str(guild.id)
    print(f"🗑️ Bot removed from {guild.name}. Cleaning up.")
    ImageEngine.cleanup_old_files(guild_id)
    if guild_id in settings:
        del settings[guild_id]
        save_settings(settings)

# --- SLASH COMMANDS ---

@bot.tree.command(name="setchannel", description="Admin: Set the channel for framing")
@app_commands.checks.has_permissions(administrator=True)
async def setchannel(
    interaction: discord.Interaction, 
    channel: Union[discord.TextChannel, discord.VoiceChannel, discord.ForumChannel, discord.Thread, discord.StageChannel]
):
    guild_id = str(interaction.guild_id)
    
    # Permission Check
    permissions = channel.permissions_for(interaction.guild.me)
    if not permissions.view_channel:
        await interaction.response.send_message(
            f"❌ **Permission Error:** I cannot access {channel.mention}.\n"
            "Please go to Channel Settings -> Permissions and give my Role **'View Channel'** permission.",
            ephemeral=True
        )
        return

    if guild_id not in settings:
        settings[guild_id] = {}

    settings[guild_id]["target_channel_id"] = channel.id
    save_settings(settings)
    await interaction.response.send_message(f"✅ Channel set to {channel.mention} for this server.")

@setchannel.error
async def setchannel_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.TransformerError):
        await interaction.response.send_message(
            f"❌ **Selection Error:** I couldn't find that channel. Please click it from the list instead of typing.",
            ephemeral=True
        )
    else:
        await interaction.response.send_message(f"❌ Error: {error}", ephemeral=True)

@bot.tree.command(name="uploadframe", description="Admin: Upload the transparent PNG frame")
@app_commands.checks.has_permissions(administrator=True)
async def uploadframe(interaction: discord.Interaction, attachment: discord.Attachment):
    if not attachment.filename.lower().endswith(".png"):
        return await interaction.response.send_message("❌ Error: Frame must be a PNG.", ephemeral=True)
    
    await interaction.response.defer()
    guild_id = str(interaction.guild_id)

    try:
        ImageEngine.cleanup_old_files(guild_id)
        
        frame_path, _ = ImageEngine.get_paths(guild_id)
        await attachment.save(frame_path)
        
        bbox = ImageEngine.analyze_frame_and_save(guild_id)
        
        await interaction.followup.send(f"🖼️ New frame uploaded!")
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")

# --- MESSAGE LISTENER ---
@bot.event
async def on_message(message):
    if message.author.bot: return
    
    guild_id = str(message.guild.id)
    if guild_id not in settings: return
    
    target_channel = settings[guild_id].get("target_channel_id")
    if message.channel.id != target_channel: return
    
    if not message.attachments: return

    frame_path, _ = ImageEngine.get_paths(guild_id)
    if not os.path.exists(frame_path): return

    att = message.attachments[0]
    if att.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        loading_msg = await message.channel.send("🎨 composing...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(att.url) as resp:
                    user_img_data = await resp.read()
                    
                    final_result = await bot.loop.run_in_executor(None, ImageEngine.process_final_image, user_img_data, guild_id)
                    
                    if final_result:
                        with io.BytesIO() as binary:
                            final_result.save(binary, 'PNG')
                            binary.seek(0)
                            await message.reply(
                                content=message.author.mention, 
                                file=discord.File(fp=binary, filename='framed.png')
                            )
            await loading_msg.delete()
        except Exception as e:
            await message.channel.send(f"❌ Error: {e}")
            print(f"Error in {message.guild.name}: {e}")

if __name__ == "__main__":
    bot.run(config.TOKEN)