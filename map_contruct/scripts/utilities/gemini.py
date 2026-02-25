# python bag_gemini_labeler.py --bag hallway_ros2/ --topic /camera/image --stride 3 --out labels.csv

#api key - AIzaSyAqX1loin2TTl2BbYDW1fd88m3acLJ1IV8
API_KEY = "AIzaSyAqX1loin2TTl2BbYDW1fd88m3acLJ1IV8"

#!/usr/bin/env python3
import os, sys, json, csv, signal
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import google.generativeai as genai

# ---------------- Gemini ----------------
if not API_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not set")
genai.configure(api_key=API_KEY)

def classify_image(jpeg_bytes: bytes, model_name="gemini-1.5-flash") -> int:
    """
    Returns 1 if open, 0 if blocked.
    Forces JSON-only output for robustness.
    """
    model = genai.GenerativeModel(model_name)
    prompt = (
        "Tell the given image is open or blocked. "
        "Return JSON ONLY as {\"label\":1} for open and {\"label\":0} for blocked. "
        "No extra keys or text."
    )
    resp = model.generate_content(
        [prompt, {"mime_type": "image/jpeg", "data": jpeg_bytes}],
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    data = json.loads(resp.text)
    lab = int(data["label"])
    if lab not in (0, 1):
        raise ValueError(f"Invalid label {lab}")
    return lab

def to_jpeg_bytes(img_bgr, q=90) -> bytes:
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return enc.tobytes()

# ------------- Node ---------------------
class GeminiCSVNode(Node):
    def __init__(self,
                 topic="/argus/ar0234_front_left/image_raw",
                 compressed=False,
                 stride=1,
                 model_name="gemini-1.5-flash",
                 out_csv="labels.csv",
                 jpeg_quality=90):
        super().__init__("gemini_csv_node")
        self.bridge = CvBridge()
        self.model_name = model_name
        self.stride = max(1, int(stride))
        self.jpeg_q = int(jpeg_quality)
        self.counter = 0

        # CSV setup
        self.out_csv = out_csv
        self.csv_file = open(self.out_csv, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "label"])  # 0=blocked, 1=open
        self.get_logger().info(f"Writing to {self.out_csv}")

        # Sub
        if compressed:
            self.sub = self.create_subscription(CompressedImage, topic, self.cb_compressed, 10)
            self.get_logger().info(f"Subscribed (CompressedImage) to {topic}")
        else:
            self.sub = self.create_subscription(Image, topic, self.cb_raw, 10)
            self.get_logger().info(f"Subscribed (Image) to {topic}")

        # Handle Ctrl+C cleanly so CSV is flushed
        signal.signal(signal.SIGINT, self._signal_handler)

    # --- Callbacks ---
    def cb_raw(self, msg: Image):
        self.counter += 1
        if self.counter % self.stride != 0:
            return
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._classify_and_log(msg.header.stamp.sec, msg.header.stamp.nanosec, img_bgr)
        except Exception as e:
            self.get_logger().error(f"Raw image handling failed: {e}")

    def cb_compressed(self, msg: CompressedImage):
        self.counter += 1
        if self.counter % self.stride != 0:
            return
        try:
            img_bgr = cv2.imdecode(
                np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            self._classify_and_log(msg.header.stamp.sec, msg.header.stamp.nanosec, img_bgr)
        except Exception as e:
            self.get_logger().error(f"Compressed image handling failed: {e}")

    def _classify_and_log(self, sec: int, nsec: int, img_bgr):
        ts = f"{sec}.{nsec:09d}"
        jpg = to_jpeg_bytes(img_bgr, q=self.jpeg_q)
        label = classify_image(jpg, model_name=self.model_name)
        # 0=blocked, 1=open
        self.csv_writer.writerow([ts, label])
        self.csv_file.flush()
        self.get_logger().info(f"{ts} -> {label}")

    def _signal_handler(self, *_):
        self.get_logger().info("Shutting down; closing CSV.")
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        rclpy.shutdown()

def main():
    import argparse
    import numpy as np  # needed if you use compressed callback
    parser = argparse.ArgumentParser(description="Subscribe to a ROS2 image topic and log Gemini open/blocked (0/1) to CSV.")
    parser.add_argument("--topic", default="/argus/ar0234_front_left/image_raw")
    parser.add_argument("--compressed", action="store_true", help="Use if your topic is sensor_msgs/CompressedImage")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--model", default="gemini-1.5-flash")
    parser.add_argument("--out", default="labels.csv")
    parser.add_argument("--jpegq", type=int, default=90)
    args = parser.parse_args()

    rclpy.init()
    node = GeminiCSVNode(
        topic=args.topic,
        compressed=args.compressed,
        stride=args.stride,
        model_name=args.model,
        out_csv=args.out,
        jpeg_quality=args.jpegq,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._signal_handler()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
