import argparse
from pathlib import Path

from PIL import Image
from psgan import Inference
from fire import Fire
import numpy as np

import faceutils as futils
from psgan import PostProcess
from setup import setup_config, setup_argparser
import cv2


# def main(save_path='transferred_image.png'):
def main():
    parser = setup_argparser()
    # parser.add_argument(
    #     "--source_path",
    #     default="./assets/images/non-makeup/xfsy_0106.png",
    #     metavar="FILE",
    #     help="path to source image")
    parser.add_argument(
        "--reference_dir",
        default="assets/images/makeup",
        help="path to reference images")
    parser.add_argument(
        "--speed",
        action="store_true",
        help="test speed")
    parser.add_argument(
        "--device",
        default="cpu",
        help="device used for inference")
    parser.add_argument(
        "--model_path",
        default="assets/models/G.pth",
        help="model for loading")
    parser.add_argument(
        "--video_path",
        default="C:/Users/babymlin/Videos/iVCam/mp4.mp4",
        help="path to source video")
    
    
    args = parser.parse_args()
    config = setup_config(args)

    # Using the second cpu
    inference = Inference(
        config, args.device, args.model_path)
    postprocess = PostProcess(config)

    reference_paths = list(Path(args.reference_dir).glob("*"))
    np.random.shuffle(reference_paths)
    
    # if args.source_path :
    #     print("轉換圖片...")
    #     for reference_path in reference_paths:
    #         if not reference_path.is_file():
    #             print(reference_path, "is not a valid file.")
    #             continue
    #         save_path = args.source_path.split("/")[-1].split(".")[0] + "_psgan.png"
    #         source = Image.open(args.source_path).convert("RGB")
    #         reference = Image.open(reference_path).convert("RGB")
    
    #         # Transfer the psgan from reference to source.
    #         image, face = inference.transfer(source, reference, with_face=True)
    #         source_crop = source.crop(
    #             (face.left(), face.top(), face.right(), face.bottom()))
    #         image = postprocess(source_crop, image)
    #         image.save(save_path)

    if args.video_path :
        print("讀取影片或開啟鏡頭...")
        for reference_path in reference_paths:
            if not reference_path.is_file():
                print(reference_path, "is not a valid file.")
                continue
            source = args.video_path
            if source != "0":
                print("讀取檔案...")
                cap = cv2.VideoCapture(source)
            else:
                print("開啟鏡頭...")
                cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret==True:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    source = Image.fromarray(frame_rgb)
                    reference = Image.open(reference_path).convert("RGB")
                    image, face = inference.transfer(source, reference, with_face=True)
                    source_crop = source.crop((face.left(), face.top(), face.right(), face.bottom()))
                    image = postprocess(source_crop, image)
                    result = np.array(image, np.uint8)
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Viedo", result)
                if cv2.waitKey(2000) != -1:
                    break
            cap.release()
            cv2.destroyAllWindows()                
        
    if args.speed:
        import time
        start = time.time()
        for _ in range(100):
            inference.transfer(source, reference)
        print("Time cost for 100 iters: ", time.time() - start)


if __name__ == '__main__':
    main()
