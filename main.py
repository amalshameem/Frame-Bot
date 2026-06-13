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
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

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

# --- LOAD AI MODELS ---

print("⏳ Loading YOLO Model...")
model = YOLO("yolo11s.pt") 
print("✅ YOLO Model Loaded.")

def load_birefnet_model():
    print("⏳ Loading BiRefNet Model (ZhengPeng7/BiRefNet)...")
    model_bi = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
    
    # Identify device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    model_bi.to(device)
    if device == 'cpu':
        model_bi.float() # Ensure float32 on CPU to avoid Half/float mismatch
        
    model_bi.eval()
    print(f"✅ BiRefNet Loaded on {device}.")
    return model_bi, device

birefnet_model, birefnet_device = load_birefnet_model()

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
    user_adjustments = {}
    user_selected_frame = {}

    @staticmethod
    def get_paths(guild_id, slot):
        return (
            os.path.join(FRAMES_DIR, f"{guild_id}_frame_{slot}.png"),
            os.path.join(FRAMES_DIR, f"{guild_id}_mask_{slot}.png")
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
    def analyze_frame_and_save(guild_id, slot):
        frame_path, mask_path = ImageEngine.get_paths(guild_id, slot)
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

        if guild_id not in ImageEngine.state:
            ImageEngine.state[guild_id] = {}
            
        ImageEngine.state[guild_id][str(slot)] = {
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
    def remove_bg(image):
        img_rgb = image.convert('RGB')
        original_size = img_rgb.size
        
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Get dtype from model
        dtype = next(birefnet_model.parameters()).dtype
        input_tensor = transform_image(img_rgb).unsqueeze(0).to(birefnet_device).to(dtype)
        
        with torch.no_grad():
            preds = birefnet_model(input_tensor)[-1].sigmoid().cpu()
        
        mask = preds[0].squeeze()
        mask_pil = transforms.ToPILImage()(mask).resize(original_size)
        
        birefnet_result = img_rgb.copy()
        birefnet_result.putalpha(mask_pil)
        return birefnet_result

    @staticmethod
    def process_final_image(user_img, guild_id, slot, x_off=0, y_off=0, zoom=1.0):
        slot = str(slot)
        if guild_id not in ImageEngine.state or slot not in ImageEngine.state[guild_id]:
            ImageEngine.analyze_frame_and_save(guild_id, slot)

        guild_data = ImageEngine.state[guild_id][slot]
        # user_img is now expected to be a PIL Image (RGBA)
        frame_img = guild_data["current_frame"]
        mask_img = guild_data["current_mask"]
        hx, hy, hw, hh = guild_data["frame_hole_bbox"]
        
        # 1. Canvas Dimensions
        canvas_w, canvas_h = frame_img.size
        
        # 2. Use AI to find the subject center (Smart Crop Logic)
        # Instead of cropping, we use these coordinates to "aim" the image
        cx, cy = ImageEngine.detect_subject_center(user_img)
        
        # 3. Scale User Image to fit the hole height/width initially
        img_w, img_h = user_img.size
        ratio = max(hw / img_w, hh / img_h) # Use max to ensure the hole is filled
        
        new_w = int(img_w * ratio * zoom)
        new_h = int(img_h * ratio * zoom)
        resized_user = user_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 4. Calculate Smart Center
        # We map the AI detected center to the new resized dimensions
        scale_x, scale_y = new_w / img_w, new_h / img_h
        smart_cx, smart_cy = int(cx * scale_x), int(cy * scale_y)
        
        # Determine the top-left coordinate to place the image so that 
        # the 'smart center' is in the middle of the frame hole
        hole_center_x = hx + (hw // 2)
        hole_center_y = hy + (hh // 2)
        
        start_x = hole_center_x - smart_cx
        start_y = hole_center_y - smart_cy
        
        # 5. Create Background Layer
        background = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        # Paste with Smart Start + User Manual Adjustments
        background.paste(resized_user, (start_x + x_off, start_y + y_off))
        
        # 6. Apply Mask and Frame Overlay
        final_composite = Image.composite(background, Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0)), mask_img)
        final_composite.alpha_composite(frame_img)
        
        return final_composite
    
    @staticmethod
    def generate_preview(guild_id):
        frames = []
        for slot in range(1, 6):
            frame_path, _ = ImageEngine.get_paths(guild_id, slot)
            if os.path.exists(frame_path):
                # Load frame and add a label/border so users know which is which
                img = Image.open(frame_path).convert("RGBA")
                # Resize for preview (e.g., max 300px height)
                img.thumbnail((300, 300))
                frames.append((slot, img))
        
        if not frames:
            return None

        # Create a canvas to hold all frames side-by-side
        total_width = sum(f[1].width for f in frames) + (len(frames) * 10)
        max_height = max(f[1].height for f in frames) + 40 # Space for slot labels
        
        preview_canvas = Image.new("RGBA", (total_width, max_height), (0, 0, 0, 0))
        
        current_x = 0
        for slot, img in frames:
            # Paste frame
            preview_canvas.paste(img, (current_x, 40), img)
            # (Optional) You could use ImageDraw here to text labels "Slot 1", etc.
            current_x += img.width + 10
            
        return preview_canvas

class OpenEditorView(discord.ui.View):
    def __init__(self, user_img_url, guild_id, slot, user_id):
        super().__init__(timeout=300)
        self.user_img_url = user_img_url
        self.guild_id = guild_id
        self.slot = slot
        self.user_id = user_id

    @discord.ui.button(label="🎨 Open Private Editor", style=discord.ButtonStyle.success)
    async def open_editor(self, interaction: discord.Interaction, button: discord.ui.Button):
        # This starts the ephemeral (private) adjustment session
        view = AdjustView(self.user_img_url, self.guild_id, self.slot, self.user_id)
        await view.update_image(interaction)
        # Delete the trigger message to keep channel clean
        await interaction.message.delete()

class AdjustView(discord.ui.View):
    def __init__(self, user_img_url, guild_id, slot, user_id):
        # We set the timeout to 600 seconds (10 minutes)
        super().__init__(timeout=600) 
        self.user_img_url = user_img_url
        self.guild_id = guild_id
        self.slot = slot
        self.user_id = user_id
        self.original_img = None
        self.bg_removed_img = None
        self.is_bg_removed = False
        
        if user_id not in ImageEngine.user_adjustments:
            ImageEngine.user_adjustments[user_id] = {"x": 0, "y": 0, "zoom": 1.0}

    async def on_timeout(self):
        """Automatically called when the user stops interacting for 10 minutes"""
        if self.user_id in ImageEngine.user_adjustments:
            del ImageEngine.user_adjustments[self.user_id]
        
        # Optional: Log the timeout for debugging
        print(f"⏱️ Session for User {self.user_id} timed out and was cleared.")

    async def load_image(self):
        if self.original_img is None:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.user_img_url) as resp:
                    data = await resp.read()
                    self.original_img = Image.open(io.BytesIO(data)).convert("RGBA")

    async def update_image(self, interaction: discord.Interaction):
        await self.load_image()
        
        # Every time the user clicks a button, the 10-minute timer resets automatically.
        adj = ImageEngine.user_adjustments[self.user_id]
        
        current_img = self.bg_removed_img if self.is_bg_removed else self.original_img
        
        final = await bot.loop.run_in_executor(
            None, ImageEngine.process_final_image, current_img, self.guild_id, self.slot, adj["x"], adj["y"], adj["zoom"]
        )
        
        with io.BytesIO() as out:
            final.save(out, 'PNG')
            out.seek(0)
            file = discord.File(fp=out, filename="adjust.png")
            
            if interaction.response.is_done():
                await interaction.edit_original_response(
                    content=f"🛠️ **Editor (Slot {self.slot}):**\n*This session will expire in 10 mins of inactivity.*",
                    attachments=[file], 
                    view=self
                )
            else:
                await interaction.response.send_message(
                    content=f"🛠️ **Editor (Slot {self.slot}):**", 
                    file=file, view=self, ephemeral=True
                )

    async def initial_send(self, message: discord.Message):
        await self.load_image()
        adj = ImageEngine.user_adjustments[self.user_id]
        current_img = self.bg_removed_img if self.is_bg_removed else self.original_img
        
        final = await bot.loop.run_in_executor(
            None, ImageEngine.process_final_image, current_img, self.guild_id, self.slot, adj["x"], adj["y"], adj["zoom"]
        )
        
        with io.BytesIO() as out:
            final.save(out, 'PNG')
            out.seek(0)
            file = discord.File(fp=out, filename="adjust.png")
            await message.channel.send(
                content=f"🛠️ **Editor (Slot {self.slot}):**", 
                file=file, view=self
            )

    @discord.ui.button(label="✂️ Remove BG", style=discord.ButtonStyle.primary, row=0)
    async def remove_bg_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        if self.bg_removed_img is None:
            await self.load_image()
            self.bg_removed_img = await bot.loop.run_in_executor(
                None, ImageEngine.remove_bg, self.original_img
            )
        self.is_bg_removed = True
        await self.update_image(interaction)

    @discord.ui.button(label="🔄 Restore BG", style=discord.ButtonStyle.secondary, row=0)
    async def restore_bg_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.is_bg_removed = False
        await self.update_image(interaction)

    @discord.ui.button(label="➕ Zoom", style=discord.ButtonStyle.success, row=1)
    async def zoom_in(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        ImageEngine.user_adjustments[self.user_id]["zoom"] += 0.25
        await self.update_image(interaction)

    @discord.ui.button(label="⬆️ Up", style=discord.ButtonStyle.secondary, row=1)
    async def up(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        ImageEngine.user_adjustments[self.user_id]["y"] -= 40
        await self.update_image(interaction)

    @discord.ui.button(label="➖ Zoom", style=discord.ButtonStyle.danger, row=1)
    async def zoom_out(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        ImageEngine.user_adjustments[self.user_id]["zoom"] = max(0.5, ImageEngine.user_adjustments[self.user_id]["zoom"] - 0.1)
        await self.update_image(interaction)

    @discord.ui.button(label="⬅️ Left", style=discord.ButtonStyle.secondary, row=2)
    async def left(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        ImageEngine.user_adjustments[self.user_id]["x"] -= 40
        await self.update_image(interaction)

    @discord.ui.button(label="⬇️ Down", style=discord.ButtonStyle.secondary, row=2)
    async def down(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        ImageEngine.user_adjustments[self.user_id]["y"] += 40
        await self.update_image(interaction)

    @discord.ui.button(label="➡️ Right", style=discord.ButtonStyle.secondary, row=2)
    async def right(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        ImageEngine.user_adjustments[self.user_id]["x"] += 40
        await self.update_image(interaction)
    
    @discord.ui.button(label="✅ Finish", style=discord.ButtonStyle.primary, row=3)
    async def finish(self, interaction: discord.Interaction, button: discord.ui.Button):
        # 1. Inform the user the process is complete
        await interaction.response.edit_message(content="✨ **Saving...**", view=None, attachments=[])
        
        # 2. Generate the final image one last time
        adj = ImageEngine.user_adjustments[self.user_id]
        
        await self.load_image()
        current_img = self.bg_removed_img if self.is_bg_removed else self.original_img
        
        final = await bot.loop.run_in_executor(
            None, ImageEngine.process_final_image, current_img, self.guild_id, self.slot, adj["x"], adj["y"], adj["zoom"]
        )
        
        with io.BytesIO() as out:
            final.save(out, 'PNG')
            out.seek(0)
            file = discord.File(fp=out, filename="final_framed.png")
            
            # 3. Send to the public channel (NOT ephemeral)
            target_channel_id = settings[self.guild_id].get("target_channel_id")
            channel = bot.get_channel(target_channel_id)
            
            if channel:
                await channel.send(
                    content=f"🎨 **New Image Created by {interaction.user.mention}**",
                    file=file
                )

        # 4. Clean up memory
        if self.user_id in ImageEngine.user_adjustments:
            del ImageEngine.user_adjustments[self.user_id]

# --- BOT SETUP ---
class FrameBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="?", intents=intents)

    async def setup_hook(self):
        await self.tree.sync()

bot = FrameBot()

@bot.event
async def on_connect():
    print(f"📡 Bot connected to Discord (Latency: {round(bot.latency * 1000)}ms).")

@bot.event
async def on_ready():
    print(f"✅ Bot is ONLINE as {bot.user} (ID: {bot.user.id})")
    print(f"🔗 Guilds: {len(bot.guilds)}")

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

@bot.tree.command(name="uploadframe", description="Admin: Upload a frame to a specific slot")
@app_commands.checks.has_permissions(administrator=True)
@app_commands.describe(slot="Choose a slot (1-5)")
async def uploadframe(interaction: discord.Interaction, attachment: discord.Attachment, slot: app_commands.Range[int, 1, 5]):
    if not attachment.filename.lower().endswith(".png"):
        return await interaction.response.send_message("❌ Error: Frame must be a PNG.", ephemeral=True)
    
    await interaction.response.defer()
    guild_id = str(interaction.guild_id)

    try:
        frame_path, _ = ImageEngine.get_paths(guild_id, slot)
        await attachment.save(frame_path)
        
        ImageEngine.analyze_frame_and_save(guild_id, slot)
        
        await interaction.followup.send(f"✅ Frame slot **{slot}** updated!")
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")

@bot.tree.command(name="removeframe", description="Admin: Delete a frame from a specific slot")
@app_commands.checks.has_permissions(administrator=True)
async def removeframe(interaction: discord.Interaction, slot: app_commands.Range[int, 1, 5]):
    await interaction.response.defer(ephemeral=True)
    guild_id = str(interaction.guild_id)
    
    frame_path, mask_path = ImageEngine.get_paths(guild_id, slot)
    if os.path.exists(frame_path):
        os.remove(frame_path)
        if os.path.exists(mask_path): os.remove(mask_path)
        
        # Clear from memory so it's gone from the preview
        if guild_id in ImageEngine.state and str(slot) in ImageEngine.state[guild_id]:
            del ImageEngine.state[guild_id][str(slot)]
            
        await interaction.followup.send(f"🗑️ Slot {slot} removed.")
    else:
        await interaction.followup.send("⚠️ That slot is already empty.")

@bot.tree.command(name="frames", description="Show available frames to choose from")
async def show_frames(interaction: discord.Interaction):
    guild_id = str(interaction.guild_id)
    preview_img = await bot.loop.run_in_executor(None, ImageEngine.generate_preview, guild_id) #
    
    if not preview_img:
        return await interaction.response.send_message("❌ No frames available yet!", ephemeral=True) #

    with io.BytesIO() as bio:
        preview_img.save(bio, 'PNG')
        bio.seek(0)
        view = FramePicker(None, guild_id) # URL is None because they haven't uploaded yet
        await interaction.response.send_message(
            content="🎨 **Choose a frame to start:**",
            file=discord.File(bio, "preview.png"),
            view=view
        ) #

@bot.tree.command(name="useframe", description="Upload your photo to a specific frame slot")
@app_commands.describe(slot="Choose a slot (1-5)", image="The photo you want to frame")
async def useframe(interaction: discord.Interaction, slot: app_commands.Range[int, 1, 5], image: discord.Attachment):
    # Defer ephemerally so ONLY the user sees the loading state
    await interaction.response.defer(ephemeral=True)
    
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        return await interaction.followup.send("❌ Please upload a valid image file.")

    guild_id = str(interaction.guild_id)
    user_id = interaction.user.id

    # Check if the frame exists
    frame_path, _ = ImageEngine.get_paths(guild_id, slot)
    if not os.path.exists(frame_path):
        return await interaction.followup.send(f"❌ Slot {slot} has no frame uploaded by an admin.")

    # Initialize the editor view immediately
    view = AdjustView(image.url, guild_id, slot, user_id)
    
    # Process the initial image and send it as an ephemeral followup
    # This automatically "opens" the editor for the user
    await view.update_image(interaction)

# --- MESSAGE LISTENER ---
class FramePicker(discord.ui.View):
    def __init__(self, user_img_url, guild_id):
        super().__init__(timeout=300)
        self.user_img_url = user_img_url
        self.guild_id = str(guild_id)
        
        # Dynamically create buttons for slots that actually have frames
        for slot in range(1, 6):
            frame_path, _ = ImageEngine.get_paths(self.guild_id, slot)
            if os.path.exists(frame_path):
                btn = discord.ui.Button(label=f"Frame {slot}", custom_id=str(slot), style=discord.ButtonStyle.primary)
                btn.callback = self.button_callback
                self.add_item(btn)

    async def on_timeout(self):
        # If the user doesn't pick a frame, we don't need to do much,
        # but you could clear their selection if you were tracking it here.
        pass

    async def button_callback(self, interaction: discord.Interaction):
        slot = interaction.data['custom_id']
        # Guide the user to the ephemeral command
        await interaction.response.send_message(
            content=f"✅ **Frame {slot} selected!**\n Use the command: `/useframe slot:{slot}` and attach your photo.",
            ephemeral=True
        )

@bot.event
async def on_message(message):
    if message.author.bot or not message.attachments: return

    user_id = message.author.id
    if user_id not in ImageEngine.user_selected_frame: return

    slot = ImageEngine.user_selected_frame[user_id]
    att = message.attachments[0]

    if att.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        # Clear selection so they must pick a frame again for a new photo
        del ImageEngine.user_selected_frame[user_id]

        view = AdjustView(att.url, str(message.guild.id), slot, user_id)
        await view.initial_send(message)

if __name__ == "__main__":
    print("🚀 Starting bot...")
    bot.run(config.TOKEN)
