import os
import math
from pathlib import Path
from typing import List, Union, Literal, Optional

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline


Layout = Literal["auto_grid", "pentagon", "stitch_right"]


class Flux2Merger:

    def __init__(
        self,
        model_path="./models/FLUX.2-klein-base-9B",
        *,
        device="cuda",
        dtype=torch.bfloat16,
        megapixel=1.0,
        divisible_by=64,
        steps=50,
        guidance_scale=4.0,
        seed=0,
        layout: Layout = "pentagon",
        grid_cell_height=640,
        spacing=0,
        pad_color=(255, 255, 255),
    ):

        # ===== MODEL LOAD (YOUR EXACT STYLE) =====
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True
        ).to(device)

        self.device = device
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.layout = layout
        self.grid_cell_height = grid_cell_height
        self.spacing = spacing
        self.pad_color = pad_color

        s = int(math.sqrt(int(megapixel * 1_000_000)))
        s -= s % divisible_by
        self.target_w = self.target_h = max(divisible_by, s)

        print(f"[Flux2Merger] Ready | target={self.target_w}x{self.target_h}")

    # =========================================================
    # IMAGE HELPERS
    # =========================================================
    @staticmethod
    def _load_rgb(p):
        return Image.open(str(p)).convert("RGB")

    # ---------- stitch right ----------
    def _stitch_right(self, paths):

        imgs = [self._load_rgb(p) for p in paths]
        H = max(i.height for i in imgs)
        W = sum(i.width for i in imgs)

        canvas = Image.new("RGB", (W, H), self.pad_color)

        x = 0
        for im in imgs:
            canvas.paste(im, (x, (H - im.height)//2))
            x += im.width

        return canvas

    # ---------- auto grid ----------
    def _grid_auto(self, paths):

        imgs = [self._load_rgb(p) for p in paths]
        n = len(imgs)
        cols = int(math.ceil(math.sqrt(n)))
        rows = (n + cols - 1)//cols

        cell_h = self.grid_cell_height
        scaled = []

        for im in imgs:
            ar = im.width/im.height
            scaled.append(im.resize((int(cell_h*ar), cell_h)))

        row_imgs = [scaled[r*cols:(r+1)*cols] for r in range(rows)]

        W = max(sum(im.width for im in row) for row in row_imgs)
        H = sum(max(im.height for im in row) for row in row_imgs)

        canvas = Image.new("RGB", (W, H), self.pad_color)

        y = 0
        for row in row_imgs:
            x = 0
            rh = max(im.height for im in row)
            for im in row:
                canvas.paste(im, (x, y))
                x += im.width
            y += rh

        return canvas

    # ---------- pentagon layout (BEST for your 5 prompts) ----------
    def _pentagon_layout(self, paths):

        imgs = [self._load_rgb(p) for p in paths]

        center = imgs[0]
        others = imgs[1:]

        center_size = 420
        corner_size = int(center_size * 0.6)

        center = center.resize((center_size, center_size))
        others = [i.resize((corner_size, corner_size)) for i in others]

        canvas_size = center_size + 2*corner_size + 40
        canvas = Image.new("RGB", (canvas_size, canvas_size), self.pad_color)

        cx = (canvas_size - center_size)//2
        cy = (canvas_size - center_size)//2

        canvas.paste(center, (cx, cy))

        pos = [
            (10,10),
            (canvas_size-corner_size-10,10),
            (10,canvas_size-corner_size-10),
            (canvas_size-corner_size-10,canvas_size-corner_size-10),
        ]

        for im,p in zip(others,pos):
            canvas.paste(im,p)

        return canvas

    # =========================================================
    # CONTEXT PREP
    # =========================================================
    def prepare_context(self, image_paths):

        if self.layout == "stitch_right":
            stitched = self._stitch_right(image_paths)

        elif self.layout == "pentagon":
            stitched = self._pentagon_layout(image_paths)

        else:
            stitched = self._grid_auto(image_paths)

        # ===== FORCE SQUARE =====
        sw, sh = stitched.size
        s = max(self.target_w/sw, self.target_h/sh)

        resized = stitched.resize(
            (int(sw*s), int(sh*s)),
            Image.LANCZOS
        )

        left = (resized.width-self.target_w)//2
        top  = (resized.height-self.target_h)//2

        return resized.crop((left, top, left+self.target_w, top+self.target_h))

    # =========================================================
    # GENERATE (MERGE)
    # =========================================================
    def generate(
        self,
        image_paths: List[Union[str, Path]],
        *,
        prompt: str = "",
        out_path: Optional[Union[str, Path]] = None,
    ):

        context = self.prepare_context(image_paths)

        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        result = self.pipe(
            image=context,
            prompt=prompt,
            height=self.target_h,
            width=self.target_w,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.steps,
            generator=generator
        )

        img = result.images[0]

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            print(f"[Flux2Merger] Saved → {out_path}")

        return img