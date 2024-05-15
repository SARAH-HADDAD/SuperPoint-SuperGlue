import time
import numpy as np
import onnxruntime
import cv2
import torch

def spsg_model_to_onnx(sp_model_path, sg_model_path, image1_path, image2_path, width, height, max_keypoint):
    """
    Converts a SuperPoint and SuperGlue (SPSG) model to ONNX format.
    """
    from models.sgsp import SPSG
    input1, img1, tensor1 = process_in(image1_path, width, height)
    input2, img2, tensor2 = process_in(image2_path, width, height)
    model = SPSG(sp_model_path, sg_model_path, width, height, max_keypoint)
    input_names = ["input1", "input2"]
    output_names = ['mkpts0', 'mkpts1', 'confidence']
    dummy_input = (tensor1, tensor2)
    mkpts0_th, mkpts1_th, confidence_th = model(tensor1, tensor2)
    ONNX_name = "sgsg.onnx"
    torch.onnx.export(model.eval(), dummy_input, ONNX_name,
                      verbose=True,
                      input_names=input_names, opset_version=16,
                      dynamic_axes={
                          'input1': {2: 'image_height', 3: "image_width"},
                          'input2': {2: 'image_height', 3: "image_width"},
                      },
                      output_names=output_names)
    spsg_model_sess = onnxruntime.InferenceSession(ONNX_name, providers=['CUDAExecutionProvider'])
    input_dic = {}
    input_list = [tensor1.numpy(), tensor2.numpy()]
    for i in range(len(spsg_model_sess.get_inputs())):
        input_dic[spsg_model_sess.get_inputs()[i].name] = input_list[i]
    mkpts0, mkpts1, confidence = spsg_model_sess.run(None, input_dic)
    mkpts0_th = mkpts0_th.detach().numpy().astype(mkpts0.dtype)

def process_in(image_path, width, height):
    """
    Reads an image from a given path, resizes it, converts it to grayscale, and normalizes it.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (width, height))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    input = np.expand_dims(np.expand_dims(img_gray, 0), 0).astype(np.float32)
    return input, img, torch.from_numpy(input)

def sgsp_torch_predict(sp_model_path, sg_model_path, image1_path, image2_path, width, height, max_keypoint, threshold=0.9):
    """
    Uses the SPSG model to make predictions on two images.
    """
    from models.sgsp import SPSG
    input1, img1, tensor1 = process_in(image1_path, width, height)
    input2, img2, tensor2 = process_in(image2_path, width, height)
    model = SPSG(sp_model_path, sg_model_path, width, height, max_keypoint)
    mkpts0, mkpts1, confidence = model(tensor1, tensor2)
    img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1] + 10, 3))
    img[:, :img1.shape[1], ] = img1
    img[:, -img1.shape[1]:] = img2
    img = img.astype(np.uint8)
    mkpts0 = mkpts0.to('cpu').numpy().astype(np.int32)
    mkpts1 = mkpts1.to('cpu').numpy().astype(np.int32)
    confidence = confidence.detach().numpy().astype(np.float32)
    for i in range(0, mkpts0.shape[0]):
        if (confidence[i] > threshold):
            pt0 = mkpts0[i]
            pt1 = mkpts1[i]
            cv2.circle(img, (pt0[0], pt0[1]), 1, (0, 0, 255), 2)
            cv2.circle(img, (pt1[0] + img1.shape[1] + 10, pt1[1]), 1, (0, 0, 255), 2)
            cv2.line(img, (pt0[0], pt0[1]), (pt1[0] + img1.shape[1] + 10, pt1[1]), (0, 255, 0), 1)
    cv2.imshow("img_torch", img)
    cv2.waitKey(0)

def spsg_onnx_predict(spsg_model_path, image1_path, image2_path, width, height, threshold=0.9):
    """
    Uses the ONNX version of the SPSG model for prediction.
    """
    spsg_model_sess = onnxruntime.InferenceSession(spsg_model_path, providers=['CUDAExecutionProvider'])
    input1, img1, tensor1 = process_in(image1_path, width, height)
    input2, img2, tensor2 = process_in(image2_path, width, height)
    input_dic = {}
    input_list = [input1, input2]
    for i in range(len(spsg_model_sess.get_inputs())):
        input_dic[spsg_model_sess.get_inputs()[i].name] = input_list[i]
    mkpts0, mkpts1, confidence = spsg_model_sess.run(None, input_dic)
    mkpts0 = mkpts0.astype(np.int32)
    mkpts1 = mkpts1.astype(np.int32)
    img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1] + 10, 3))
    img[:, :img1.shape[1], ] = img1
    img[:, -img1.shape[1]:] = img2
    img = img.astype(np.uint8)
    for i in range(0, mkpts0.shape[0]):
        if (confidence[i] > threshold):
            pt0 = mkpts0[i]
            pt1 = mkpts1[i]
            cv2.circle(img, (pt0[0], pt0[1]), 1, (0, 0, 255), 2)
            cv2.circle(img, (pt1[0] + img1.shape[1] + 10, pt1[1]), 1, (0, 0, 255), 2)
            cv2.line(img, (pt0[0], pt0[1]), (pt1[0] + img1.shape[1] + 10, pt1[1]), (0, 255, 0), 1)
    cv2.imshow("img_onnx", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    width = 800
    height = 600
    max_keypoint = 100
    threshold = 0.50
    sp_model_path = "weights/superpoint_v1.pth"
    sg_model_path = "weights/superglue_outdoor.pth"
    image1_path = "image/image0-removebg-preview.png"
    image2_path = "image/image1-removebg-preview.png"
    spsg_model_to_onnx(sp_model_path, sg_model_path, image1_path, image2_path, width, height, max_keypoint)
    sgsp_torch_predict(sp_model_path, sg_model_path, image1_path, image2_path, width, height, max_keypoint, threshold)
    spsg_onnx_predict("sgsg.onnx", image1_path, image2_path, width, height, threshold)
