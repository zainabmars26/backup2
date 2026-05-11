import cv2
import os
import shutil
import tkinter as tk
from tkinter import messagebox, simpledialog, font
from PIL import Image, ImageTk

# --- Style Constants ---
BG_MAIN = "#f5f5f5"
BG_PANEL = "#ffffff"
FG_TEXT = "#333333"
ACCENT_BLUE = "#0078d4"
ACCENT_GREEN = "#228b22"
ACCENT_RED = "#d83b01"

class YOLOChecker:
    def __init__(self, root):
        self.root = root
        self.root.title("LabeL Locally")
        self.root.geometry("1200x900")
        self.root.configure(bg=BG_MAIN)
         
        logo = tk.PhotoImage(file=r"c:\Users\Zainab.Alawneh\Downloads\logo.png")
        self.root.iconphoto(False, logo)
        self.show_annotations = True  # Make sure this is True by default


        #self.classes = {}  
        self.classes_file = "classes.txt"
        self.load_classes_from_file()
        
        # --- State ---
        self.mode = "view"
        self.show_annotations = True
        self.image_files = []
        self.index = 0
        self.img_dir = ""
        self.label_dir = ""
        
        self.zoom_level = 1.0
        self.original_img = None
        self.start_x = self.start_y = None
        self.current_rect = None

        self.setup_ui()

    def setup_ui(self):
        self.custom_font = font.Font(family="Segoe UI", size=10)
        self.header_font = font.Font(family="Segoe UI", size=10, weight="bold")

        # Top Bar
        self.top_bar = tk.Frame(self.root, bg=BG_PANEL, bd=1, relief="flat")
        self.top_bar.pack(side="top", fill="x", padx=10, pady=10)

        tk.Label(self.top_bar, text="Images:", bg=BG_PANEL).pack(side="left", padx=5)
        self.img_entry = tk.Entry(self.top_bar, width=30, bg=BG_MAIN)
        self.img_entry.pack(side="left", padx=5)
        
        tk.Label(self.top_bar, text="Labels:", bg=BG_PANEL).pack(side="left", padx=5)
        self.label_entry = tk.Entry(self.top_bar, width=30, bg=BG_MAIN)
        self.label_entry.pack(side="left", padx=5)

        tk.Button(self.top_bar, text="Load Dataset", command=self.load_dataset, bg=ACCENT_BLUE, fg="white", relief="flat").pack(side="left", padx=10)

        # Bottom Nav
        self.nav_bar = tk.Frame(self.root, bg=BG_PANEL, bd=1, relief="solid")
        self.nav_bar.pack(side="bottom", fill="x", padx=10, pady=10)
        tk.Button(self.nav_bar, text="< Previous", command=self.prev_img, width=15).pack(side="left", padx=50, pady=10)
        tk.Button(self.nav_bar, text="Next >", command=self.next_img, width=15).pack(side="right", padx=50, pady=10)

        # Left Sidebar
        self.sidebar = tk.Frame(self.root, bg=BG_PANEL, width=200, bd=1, relief="solid")
        self.sidebar.pack(side="left", fill="y", padx=10)
        self.file_listbox = tk.Listbox(self.sidebar, bg=BG_PANEL, borderwidth=0)
        self.file_listbox.pack(expand=True, fill="both")
        self.file_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)

        # Right Panel
        self.right_panel = tk.Frame(self.root, bg=BG_PANEL, width=180, bd=1, relief="solid")
        self.right_panel.pack(side="right", fill="y", padx=10)
        self.draw_btn = self.create_side_btn("Draw Mode", lambda: self.set_mode("draw"))
        self.del_btn = self.create_side_btn("Delete Mode", lambda: self.set_mode("delete"))
        self.toggle_btn = self.create_side_btn("Hide Labels", self.toggle_annotations)
        self.create_side_btn("Reset View", self.reset_zoom, color="#777777")
        self.create_side_btn("Mark Corrupted", self.mark_corrupted, color=ACCENT_RED)

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#dcdcdc", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both", padx=5)

        # Bindings
        self.canvas.bind("<MouseWheel>", self.handle_zoom)
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<B1-Motion>", self.update_drawing)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    # def get_class_id_custom(self):
    #     dialog = tk.Toplevel(self.root)
    #     dialog.title("Select Class")
    #     dialog.geometry("350x450")
    #     dialog.configure(bg=BG_PANEL)
    #     dialog.transient(self.root)
    #     dialog.grab_set()

    #     result = {"id": None}

    #     # Header
    #     tk.Label(dialog, text="Existing Classes:", font=self.header_font, bg=BG_PANEL).pack(pady=(10, 5))
        
    #     # Listbox for existing items
    #     lb = tk.Listbox(dialog, height=8, bg="#ffffff", relief="solid", borderwidth=1)
    #     lb.pack(fill="both", expand=True, padx=20)
        
    #     # Populate listbox with hints like "0: Car"
    #     for cid in sorted(self.classes.keys(), key=lambda x: int(x) if x.isdigit() else x):
    #         lb.insert(tk.END, f"{cid}: {self.classes[cid]}")

    #     # New Entry Section
    #     entry_frame = tk.Frame(dialog, bg=BG_PANEL)
    #     entry_frame.pack(pady=10, padx=20, fill="x")

    #     tk.Label(entry_frame, text="New ID:", bg=BG_PANEL).grid(row=0, column=0, sticky="w")
    #     id_entry = tk.Entry(entry_frame, width=10)
    #     id_entry.grid(row=0, column=1, padx=5, pady=2)

    #     tk.Label(entry_frame, text="Class Name:", bg=BG_PANEL).grid(row=1, column=0, sticky="w")
    #     name_entry = tk.Entry(entry_frame)
    #     name_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
    #     entry_frame.columnconfigure(1, weight=1)

    #     def on_confirm():
    #         # Priority 1: Selection from Listbox
    #         if lb.curselection():
    #             pick = lb.get(lb.curselection()[0])
    #             result["id"] = pick.split(":")[0].strip()
    #             dialog.destroy()
            
    #         # Priority 2: New ID/Name entered manually
    #         elif id_entry.get().strip():
    #             cid = id_entry.get().strip()
    #             name = name_entry.get().strip() or "Unknown" # Default if name is empty
                
    #             # Save to internal memory and classes.txt
    #             self.classes[cid] = name
    #             self.save_classes_to_file()
                
    #             result["id"] = cid
    #             dialog.destroy()
            
    #         else:
    #             messagebox.showwarning("Input Required", "Please select a class or enter a new ID.")

    #     tk.Button(dialog, text="Confirm", command=on_confirm, bg=ACCENT_GREEN, fg="white", 
    #             width=15, relief="flat", pady=5).pack(pady=15)
        
    #     # Focus the ID entry by default
    #     id_entry.focus_set()
        
    #     self.root.wait_window(dialog)
    #     return result["id"]


    def get_class_id_custom(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Class")
        dialog.geometry("420x550")
        dialog.configure(bg=BG_PANEL)
        dialog.transient(self.root)
        dialog.grab_set()

        result = {"id": None}

        tk.Label(dialog, text="Existing Classes", font=("Segoe UI", 14, "bold"), 
                bg=BG_PANEL, fg=FG_TEXT).pack(pady=(20, 10))
        
        # --- Scrollable Container ---
        container = tk.Frame(dialog, bg="#ffffff", bd=1, relief="solid")
        container.pack(fill="both", expand=True, padx=30)

        canvas = tk.Canvas(container, bg="#ffffff", highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#ffffff")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=340)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def refresh_rows():
            # Clear current rows
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
            
            # Sort classes numerically
            sorted_keys = sorted(self.classes.keys(), key=lambda x: int(x) if x.isdigit() else x)
            
            for cid in sorted_keys:
                row = tk.Frame(scrollable_frame, bg="#ffffff", pady=2)
                row.pack(fill="x", padx=5)

                # Clickable Label (to select the class)
                lbl_text = f"{cid}: {self.classes[cid]}"
                lbl = tk.Button(row, text=lbl_text, font=("Segoe UI", 10), anchor="w",
                                bg="#ffffff", fg=FG_TEXT, relief="flat", activebackground=ACCENT_BLUE,
                                activeforeground="white", command=lambda c=cid: select_class(c))
                lbl.pack(side="left", fill="x", expand=True)

                # Delete Button beside the class
                del_btn = tk.Button(row, text="✕", font=("Segoe UI", 8, "bold"),
                                    bg="#ffffff", fg=ACCENT_RED, relief="flat",
                                    activeforeground="#ffffff", activebackground=ACCENT_RED,
                                    command=lambda c=cid: delete_entry(c))
                del_btn.pack(side="right", padx=5)
                
                # Hover effects
                row.bind("<Enter>", lambda e, r=row: r.config(bg="#f0f0f0"))
                row.bind("<Leave>", lambda e, r=row: r.config(bg="#ffffff"))

        def select_class(cid):
            result["id"] = cid
            dialog.destroy()

        def delete_entry(cid):
            if messagebox.askyesno("Delete", f"Remove class '{cid}'?"):
                del self.classes[cid]
                self.save_classes_to_file()
                refresh_rows()

        refresh_rows()

        # --- New Entry Section ---
        tk.Frame(dialog, height=1, bg="#e0e0e0").pack(fill="x", padx=30, pady=20)
        input_container = tk.Frame(dialog, bg=BG_PANEL)
        input_container.pack(fill="x", padx=30)

        label_style = {"bg": BG_PANEL, "fg": "#666666", "font": ("Segoe UI", 9, "bold")}
        entry_style = {"font": ("Segoe UI", 10), "borderwidth": 1, "relief": "solid"}

        tk.Label(input_container, text="NEW ID", **label_style).grid(row=0, column=0, sticky="w")
        id_entry = tk.Entry(input_container, width=8, **entry_style)
        id_entry.grid(row=1, column=0, sticky="w", pady=(5, 15))

        tk.Label(input_container, text="CLASS NAME", **label_style).grid(row=0, column=1, sticky="w", padx=(20, 0))
        name_entry = tk.Entry(input_container, **entry_style)
        name_entry.grid(row=1, column=1, sticky="ew", padx=(20, 0), pady=(5, 15))
        input_container.columnconfigure(1, weight=1)

        def on_confirm():
            if id_entry.get().strip():
                cid = id_entry.get().strip()
                name = name_entry.get().strip() or "New Class"
                self.classes[cid] = name
                self.save_classes_to_file()
                result["id"] = cid
                dialog.destroy()

        tk.Button(dialog, text="ADD & CONFIRM", command=on_confirm,
                bg=ACCENT_GREEN, fg="white", font=("Segoe UI", 9, "bold"),
                relief="flat", cursor="hand2", padx=20, pady=8).pack(pady=(10, 30))

        id_entry.focus_set()
        self.root.wait_window(dialog)
        return result["id"]

    def load_classes_from_file(self):
        """Loads classes from a file. If the file doesn't exist, it does nothing."""
        self.classes = {}
        if os.path.exists(self.classes_file):
            try:
                with open(self.classes_file, "r") as f:
                    for line in f:
                        if ":" in line:
                            cid, name = line.strip().split(":", 1)
                            self.classes[cid.strip()] = name.strip()
            except Exception as e:
                print(f"Error loading classes: {e}")

    def save_classes_to_file(self):
        """Saves the current dictionary to classes.txt, creating it if it doesn't exist."""
        try:
            # Sorting by ID numerically helps keep the file and UI clean
            sorted_classes = sorted(self.classes.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
            with open(self.classes_file, "w") as f:
                for cid, name in sorted_classes:
                    f.write(f"{cid}:{name}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save classes.txt: {e}")

    def create_side_btn(self, text, command, color="#f0f0f0"):
        fg = "white" if color != "#f0f0f0" else FG_TEXT
        btn = tk.Button(self.right_panel, text=text, command=command, bg=color, fg=fg, width=18, pady=10, relief="flat")
        btn.pack(pady=5, padx=10)
        return btn

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.update_display(update_listbox=False)

    def handle_zoom(self, event):
        scale = 1.1 if event.delta > 0 else 0.9
        self.zoom_level *= scale
        self.zoom_level = max(0.1, min(self.zoom_level, 10.0))
        self.update_display(update_listbox=False)

    def get_real_coords(self, canvas_x, canvas_y):
        if not self.image_files: return 0, 0

        # 1. Get the current physical size of the canvas window
        self.root.update_idletasks()
        canv_w = self.canvas.winfo_width()
        canv_h = self.canvas.winfo_height()

        # 2. Get the size of the image currently displayed on screen
        img_w = self.tk_img.width()
        img_h = self.tk_img.height()

        # 3. Calculate Centering Offsets 
        # This is the 'gray space' between the canvas edge and the image edge
        # We use the larger of the two (scrollable area vs window) to find the true center
        sr = self.canvas.cget("scrollregion").split()
        sr_w, sr_h = float(sr[2]), float(sr[3])
        
        offset_x = (sr_w - img_w) / 2
        offset_y = (sr_h - img_h) / 2

        # 4. Reverse the transformation
        # Subtract the offset first to get to image-relative coordinates, 
        # then divide by the zoom_level (which includes the fit_scale)
        
        # Calculate the base fit_scale used in update_display
        img_name = self.image_files[self.index]
        temp_img = cv2.imread(os.path.join(self.img_dir, img_name))
        h_orig, w_orig, _ = temp_img.shape
        
        fit_scale = min((canv_w - 40) / w_orig, (canv_h - 40) / h_orig)
        total_scale = fit_scale * self.zoom_level

        real_x = (canvas_x - offset_x) / total_scale
        real_y = (canvas_y - offset_y) / total_scale

        # 5. Clamp the values to the image boundaries
        real_x = max(0, min(w_orig, real_x))
        real_y = max(0, min(h_orig, real_y))

        return real_x, real_y

        
    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def load_dataset(self):
        self.img_dir = self.img_entry.get().strip()
        self.label_dir = self.label_entry.get().strip()

        # If image directory is valid but label directory is empty
        if os.path.isdir(self.img_dir) and not self.label_dir:
            # Get the parent directory and create 'labels' next to the images folder
            parent_dir = os.path.dirname(os.path.abspath(self.img_dir))
            self.label_dir = os.path.join(parent_dir, "labels")
            
            if not os.path.exists(self.label_dir):
                os.makedirs(self.label_dir)
            
            # Update the UI entry to show the new sibling path
            self.label_entry.delete(0, tk.END)
            self.label_entry.insert(0, self.label_dir)

        if os.path.isdir(self.img_dir):
            self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.file_listbox.delete(0, tk.END)
            for f in self.image_files: self.file_listbox.insert(tk.END, f)
            if self.image_files: 
                self.index = 0
                self.update_display()

    def update_display(self, update_listbox=True):
        if not self.image_files: return
        img_name = self.image_files[self.index]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        if update_listbox:
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.index)
            self.file_listbox.see(self.index)

        # 1. Load the raw image
        img = cv2.imread(img_path)
        if img is None: return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # 2. Draw Annotations on the RAW image first
        # CHANGE: We use getattr to ensure it works even if the variable isn't set yet
        show_em = getattr(self, 'show_annotations', True) 
        
        if show_em and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5: continue
                    try:
                        cls_id, x, y, bw, bh = map(float, parts)
                        
                        # Convert YOLO normalized to pixel coordinates
                        x1 = int((x - bw/2) * w)
                        y1 = int((y - bh/2) * h)
                        x2 = int((x + bw/2) * w)
                        y2 = int((y + bh/2) * h)
                        
                        # Use bright green (0, 255, 0) to make sure they are visible!
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        cls_name = self.classes.get(str(int(cls_id)), f"ID:{int(cls_id)}")
                        cv2.putText(img_rgb, cls_name, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except:
                        continue

        # 3. Handle Scaling for the Canvas (Keep your existing scaling code here...)
        self.root.update_idletasks()

        # 3. Handle Scaling for the Canvas
        self.root.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # Calculate the scale to fit the image on screen
        fit_scale = min((canvas_w - 40) / w, (canvas_h - 40) / h)
        display_scale = fit_scale * self.zoom_level
        
        new_w, new_h = int(w * display_scale), int(h * display_scale)

        # 4. Resize and Show
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        
        sr_w = max(canvas_w, new_w)
        sr_h = max(canvas_h, new_h)
        self.canvas.config(scrollregion=(0, 0, sr_w, sr_h))
        
        self.canvas.delete("all")
        self.canvas.create_image(sr_w // 2, sr_h // 2, anchor=tk.CENTER, image=self.tk_img)

    def handle_click(self, event):
    # Use canvasx/y to get coordinates relative to the scrollable area
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        real_x, real_y = self.get_real_coords(cx, cy)
        
        if self.mode == "delete": 
            self.delete_at_point(real_x, real_y)
        elif self.mode == "draw": 
            self.start_x, self.start_y = cx, cy
            self.current_rect = self.canvas.create_rectangle(cx, cy, cx, cy, outline="red", width=2)



    def start_drawing(self, event):
        self.start_x, self.start_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.current_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def update_drawing(self, event):
        if self.mode == "draw" and self.current_rect:
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def stop_drawing(self, event):
        if self.mode == "draw" and self.current_rect:
            end_x, end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            
            # Convert start and end points back to raw image pixels
            rx1, ry1 = self.get_real_coords(self.start_x, self.start_y)
            rx2, ry2 = self.get_real_coords(end_x, end_y)
            
            #cid = simpledialog.askstring("Class", "Enter Class ID:")
            cid = self.get_class_id_custom()
            if cid:
                self.save_new_box(rx1, ry1, rx2, ry2, cid)
                
            self.current_rect = None
            self.update_display(update_listbox=False)

    

    def save_new_box(self, x1, y1, x2, y2, cid):
        # Safety check: make sure we have an image and a label path
        if not self.image_files or not self.label_dir:
            return

        img_name = self.image_files[self.index]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: return
        
        h, w, _ = img.shape
        
        # Ensure directory exists before writing
        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)

        # Create the .txt filename based on the image filename
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        lp = os.path.join(self.label_dir, txt_name)
        
        # Calculate YOLO coordinates
        bw, bh = abs(x2-x1)/w, abs(y2-y1)/h
        xc, yc = (min(x1, x2) + abs(x2-x1)/2)/w, (min(y1, y2) + abs(y2-y1)/2)/h
        
        # Open in append mode (creates file if it doesn't exist)
        with open(lp, "a") as f: 
            f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    def delete_at_point(self, nx, ny):
        img = cv2.imread(os.path.join(self.img_dir, self.image_files[self.index]))
        if img is None: return
        h, w, _ = img.shape
        lp = os.path.join(self.label_dir, os.path.splitext(self.image_files[self.index])[0] + ".txt")
        if not os.path.exists(lp): return
        rem, done = [], False
        with open(lp, 'r') as f:
            for line in f:
                c, x, y, bw, bh = map(float, line.split())
                if (x-bw/2 <= nx/w <= x+bw/2) and (y-bh/2 <= ny/h <= y+bh/2) and not done: done = True
                else: rem.append(line)
        with open(lp, 'w') as f: f.writelines(rem)
        self.update_display(update_listbox=False)

    def set_mode(self, new_mode):
        self.mode = "view" if self.mode == new_mode else new_mode
        self.draw_btn.config(bg=ACCENT_GREEN if self.mode == "draw" else "#f0f0f0")
        self.del_btn.config(bg=ACCENT_RED if self.mode == "delete" else "#f0f0f0")

    def next_img(self):
        if self.image_files:
            self.index = (self.index + 1) % len(self.image_files)
            self.update_display()

    def prev_img(self):
        if self.image_files:
            self.index = (self.index - 1) % len(self.image_files)
            self.update_display()

    def on_listbox_select(self, e):
        if self.file_listbox.curselection():
            self.index = self.file_listbox.curselection()[0]
            self.update_display(update_listbox=False)

    def mark_corrupted(self):
        # (Add your shutil logic here as previously built)
        pass

    def toggle_annotations(self):
        self.show_annotations = not self.show_annotations
        # Change button text to show user the current state
        if self.show_annotations:
            self.toggle_btn.config(text="Hide Labels", bg="#f0f0f0", fg=FG_TEXT)
        else:
            self.toggle_btn.config(text="Show Labels", bg=ACCENT_BLUE, fg="white")
        
        self.update_display(update_listbox=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOChecker(root)
    root.mainloop()
