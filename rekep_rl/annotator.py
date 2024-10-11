import sys
import os
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QScrollArea, QInputDialog, QDialog, QRadioButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


class ModeSelectionDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Mode")
        self.mode = None  # Will be set to 'mode1' or 'mode2'
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        label = QLabel("Select Mode:")
        layout.addWidget(label)

        self.mode1_radio = QRadioButton("Mode 1 - Multiple objects' key points on single image")
        self.mode2_radio = QRadioButton("Mode 2 - Multiple objects' key points on different images")
        self.mode2_radio.setChecked(True)  # Default mode is Mode 2

        layout.addWidget(self.mode1_radio)
        layout.addWidget(self.mode2_radio)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        self.setLayout(layout)

    def get_mode(self):
        if self.mode1_radio.isChecked():
            return 'mode1'
        elif self.mode2_radio.isChecked():
            return 'mode2'
        else:
            return None


class ImageAnnotator(QMainWindow):
    def __init__(self, mode='mode2'):
        super().__init__()
        self.setWindowTitle("Image Annotator")
        self.resize(1280, 720)

        self.mode = mode
        self.image = None
        self.original_image = None
        self.pixmap = None
        self.points = []
        self.objects = []
        self.count = 0
        self.image_name = ''
        self.object_names = []
        self.annotations_dir = os.path.join(os.getcwd(), 'annotations')  # 'annotations' folder in current directory
        os.makedirs(self.annotations_dir, exist_ok=True)  # Create the 'annotations' directory if it doesn't exist

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.v_layout = QVBoxLayout(self.main_widget)

        # Buttons
        self.button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.next_object_button = QPushButton("Next Object")
        self.clear_button = QPushButton("Clear Current Object Annotations")
        self.save_button = QPushButton("Save All Annotations")

        self.load_button.clicked.connect(self.load_image)
        self.next_object_button.clicked.connect(self.next_object)
        self.clear_button.clicked.connect(self.clear_annotations)
        self.save_button.clicked.connect(self.save_points)

        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.next_object_button)
        self.button_layout.addWidget(self.clear_button)
        self.button_layout.addWidget(self.save_button)

        self.v_layout.addLayout(self.button_layout)

        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.image_label.mousePressEvent = self.on_click

        # Scroll area for image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)

        self.v_layout.addWidget(self.scroll_area)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image/Video", "", "Image/Video files (*.jpg *.jpeg *.png *.mp4)", options=options
        )

        if file_path:
            if file_path.endswith('.mp4'):
                vid = cv2.VideoCapture(file_path)
                success, im = vid.read()
                if not success:
                    QMessageBox.warning(self, "Warning", "Failed to load first frame from video.")
                    return
                first_frame_path = os.path.join(self.annotations_dir, 'first_frame.jpg')
                cv2.imwrite(first_frame_path, im)
                file_path = first_frame_path

            self.image = Image.open(file_path).convert('RGB')
            self.original_image = self.image.copy()
            self.image_name = os.path.basename(file_path)
            self.points = []
            self.objects = []
            self.count = 0
            self.object_names = []
            self.next_object_button.setText("Next Object")
            self.update_image()
        else:
            QMessageBox.warning(self, "Warning", "No file selected.")

    def update_image(self):
        if self.image:
            # Convert PIL Image to QPixmap
            data = self.image.tobytes("raw", "RGB")
            qimage = QImage(data, self.image.width, self.image.height, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(self.pixmap)
            self.image_label.adjustSize()

    def on_click(self, event):
        if self.image:
            x = event.pos().x()
            y = event.pos().y()
            self.points.append((y, x))  # Note: (y, x) ordering
            self.count += 1
            self.draw_point(x, y)
            self.update_image()

    def draw_point(self, x, y):
        # Draw on the image
        draw = ImageDraw.Draw(self.image)
        side_length = 20
        s = side_length // 2
        draw.rectangle((x - s, y - s, x + s, y + s), fill='black')
        draw.rectangle((x - s + 3, y - s + 3, x + s - 3, y + s - 3), fill='white')

        # Draw the text
        font = ImageFont.load_default(size=20)
        text = str(self.count)
        text_bbox = font.getbbox(text)

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x - text_width // 2
        text_y = y - text_height // 2 - 5
        draw.text((text_x, text_y), text, font=font, fill='red')

    def next_object(self):
        if self.points:
            # Prompt for object name
            object_name, ok = QInputDialog.getText(self, "Object Name", "Enter a unique name for the object you just annotated:")
            if ok and object_name:
                if object_name in self.object_names:
                    QMessageBox.warning(self, "Warning", "Object name must be unique.")
                    return
                else:
                    self.object_names.append(object_name)
                    # Save current object's annotations in memory
                    self.objects.append({
                        'name': object_name,
                        'keypoints': self.points.copy(),
                        'no_keypoints': self.count
                    })
                    # Reset points and count
                    self.points = []
                    self.count = 0
                    # Change button text to 'Next Object' if this was the first object
                    if len(self.objects) == 1:
                        self.next_object_button.setText("Next Object")
                    # Reset the image to remove annotations or not depending on the mode
                    if self.mode == 'mode2':
                        if self.original_image:
                            self.image = self.original_image.copy()
                            self.update_image()
                    elif self.mode == 'mode1':
                        # Do not reset the image; keep the annotations
                        pass
            else:
                QMessageBox.warning(self, "Warning", "Object name cannot be empty.")
        else:
            QMessageBox.information(self, "Info", "No points to save for this object.")

    def clear_annotations(self):
        # Clear only the current object's annotations
        if self.points:
            self.points = []
            self.count = 0
            if self.original_image:
                self.image = self.original_image.copy()

                # If in Mode 1 (multiple objects on the same image), we need to redraw existing objects' points
                if self.mode == 'mode1':
                    draw = ImageDraw.Draw(self.image)
                    global_count = 0
                    for obj in self.objects:
                        # Redraw all previously saved objects
                        for i, (y, x) in enumerate(obj['keypoints']):
                            global_count += 1
                            self.draw_point_on_image(draw, x, y, global_count)

                # Update the image after clearing the current annotations
                self.update_image()
            QMessageBox.information(self, "Info", "Current object's annotations have been cleared.")
        else:
            QMessageBox.information(self, "Info", "No annotations to clear for the current object.")

    def save_points(self):
        if self.points:
            # Prompt for object name
            object_name, ok = QInputDialog.getText(self, "Object Name", "Enter a unique name for the object:")
            if ok and object_name:
                if object_name in self.object_names:
                    QMessageBox.warning(self, "Warning", "Object name must be unique.")
                    return
                else:
                    self.object_names.append(object_name)
                    # Save current object's annotations in memory
                    self.objects.append({
                        'name': object_name,
                        'keypoints': self.points.copy(),
                        'no_keypoints': self.count
                    })
                    # Reset points and count for next potential object
                    self.points = []
                    self.count = 0
                    # Reset the image to remove annotations
                    if self.mode == 'mode2':
                        if self.original_image:
                            self.image = self.original_image.copy()
                            self.update_image()
                    elif self.mode == 'mode1':
                        # Do not reset the image; keep the annotations
                        pass
            else:
                QMessageBox.warning(self, "Warning", "Object name cannot be empty.")
                return  # Don't proceed to saving

        if self.objects:
            # Create the 'annotated_images' folder inside 'annotations' directory
            annotated_images_path = os.path.join(self.annotations_dir, 'annotated_images')
            os.makedirs(annotated_images_path, exist_ok=True)

            # Prepare JSON data
            json_data = {}

            if self.mode == 'mode1':
                # In Mode 1, all annotations are on the same image
                # Create a copy of the original image
                annotated_image = self.original_image.copy()
                draw = ImageDraw.Draw(annotated_image)

                # For each object, draw its keypoints
                global_count = 0
                for obj in self.objects:
                    object_name = obj['name']
                    # Draw the keypoints for this object
                    for i, (y, x) in enumerate(obj['keypoints']):
                        global_count += 1
                        self.draw_point_on_image(draw, x, y, global_count)
                    # Add to JSON data
                    json_data[object_name] = {
                        'keypoints': obj['keypoints'],
                        'no_keypoints': obj['no_keypoints']
                    }
                # Save the image
                image_filename = f'annotated_{os.path.splitext(self.image_name)[0]}.jpg'
                image_path = os.path.join(annotated_images_path, image_filename)
                annotated_image.save(image_path)
            elif self.mode == 'mode2':
                # In Mode 2, save each object's annotations as a separate image
                for obj in self.objects:
                    object_name = obj['name']
                    # Create a copy of the original image
                    annotated_image = self.original_image.copy()
                    draw = ImageDraw.Draw(annotated_image)

                    # Draw the keypoints for this object
                    for i, (y, x) in enumerate(obj['keypoints']):
                        self.draw_point_on_image(draw, x, y, i + 1)

                    # Save the image
                    image_filename = f'annotated_{object_name}.jpg'
                    image_path = os.path.join(annotated_images_path, image_filename)
                    annotated_image.save(image_path)

                    # Add to JSON data
                    json_data[object_name] = {
                        'keypoints': obj['keypoints'],
                        'no_keypoints': obj['no_keypoints']
                    }

            # Save the JSON file
            json_filename = f'{os.path.splitext(self.image_name)[0]}_annotated.json'
            json_path = os.path.join(self.annotations_dir, json_filename)
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)

            QMessageBox.information(self, "Info", f"Annotations saved in '{self.annotations_dir}'.")
        else:
            QMessageBox.information(self, "Info", "No annotations to save.")

    def draw_point_on_image(self, draw, x, y, count, object_name=None):
        side_length = 20
        s = side_length // 2
        draw.rectangle((x - s, y - s, x + s, y + s), fill='black')
        draw.rectangle((x - s + 3, y - s + 3, x + s - 3, y + s - 3), fill='white')

        # Draw the text
        font = ImageFont.load_default(size=20)
        text = str(count)
        text_bbox = font.getbbox(text)

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x - text_width // 2
        text_y = y - text_height // 2 - 5
        draw.text((text_x, text_y), text, font=font, fill='red')

        # Draw object name if provided
        if object_name:
            name_text = object_name
            name_bbox = font.getbbox(name_text)
            name_width = name_bbox[2] - name_bbox[0]
            name_height = name_bbox[3] - name_bbox[1]
            name_x = x - name_width // 2
            name_y = y + side_length // 2 + 5
            draw.text((name_x, name_y), name_text, font=font, fill='blue')


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Show mode selection dialog
    mode_dialog = ModeSelectionDialog()
    if mode_dialog.exec_() == QDialog.Accepted:
        selected_mode = mode_dialog.get_mode()
        if selected_mode:
            window = ImageAnnotator(mode=selected_mode)
            window.show()
            sys.exit(app.exec_())
        else:
            QMessageBox.warning(None, "Warning", "No mode selected.")
            sys.exit()
    else:
        sys.exit()
