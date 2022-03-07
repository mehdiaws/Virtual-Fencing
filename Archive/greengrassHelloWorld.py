# *****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
# *****************************************************
""" A sample lambda for object detection"""
from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import greengrasssdk
import datetime
import boto3
from botocore.session import Session
import speak


# BUCKET_NAME = 'shell-deeplens-images-and-coordinates'
# s3 = boto3.resource('s3')
# s3.Bucket(BUCKET_NAME).download_file('public/coordinates/1.json', '/tmp/1.json')

def write_frame1_to_s3(img):
    session = Session()
    s3 = session.create_client('s3')
    # s3 = boto3.resource('s3')
    file_name = 'frame-1' + '.jpg'
    # You can contorl the size and quality of the image
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, jpg_data = cv2.imencode('.jpg', img, encode_param)

    mac = open('/sys/class/net/mlan0/address').readline()
    # response = s3.put_object(ACL='private', Body=mac,Bucket='images-33',Key='mac.txt')
    # response = s3.put_object(ACL='private', Body=jpg_data.tostring(),Bucket='images-mehdi',Key=file_name)
    s3.put_object(Bucket='shell-deeplens-images-and-coordinates',
                  Key='public/images/camera1.jpg',
                  Body=jpg_data.tostring())
    image_url = 'https://s3-us-east-1.amazonaws.com/shell-deeplens-images-and-coordinates/public/images/camera1.jpg'
    return image_url


def write_image_to_s3(img):
    session = Session()
    s3 = session.create_client('s3')
    # s3 = boto3.resource('s3')
    file_name = 'Camera-1: ' + str(datetime.datetime.now())[:19] + '.jpg'
    # You can contorl the size and quality of the image
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, jpg_data = cv2.imencode('.jpg', img, encode_param)

    mac = open('/sys/class/net/mlan0/address').readline()
    # response = s3.put_object(ACL='private', Body=mac,Bucket='images-33',Key='mac.txt')
    # response = s3.put_object(ACL='private', Body=jpg_data.tostring(),Bucket='images-mehdi',Key=file_name)
    s3.put_object(Bucket='shell-deeplens-images-and-coordinates',
                  Key='violations/' + file_name,
                  Body=jpg_data.tostring())
    image_url = 'https://s3-us-east-1.amazonaws.com/shell-deeplens-images-and-coordinates/violations/' + file_name
    return image_url


class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """

    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p': (1920, 1080), '720p': (1280, 720), '480p': (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255 * np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()


def greengrass_infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
        # This object detection model is implemented as single shot detector (ssd), since
        # the number of labels is small we create a dictionary that will help us convert
        # the machine labels to human readable labels.
        model_type = 'ssd'
        output_map = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
                      7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'dinning table',
                      12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
                      16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train',
                      20: 'tvmonitor'}
        # Create an IoT client for sending to messages to the cloud.
        client = greengrasssdk.client('iot-data')
        ### Jemmy
        iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
        # iot_topic = 'deeplensTopic'
        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()
        # The sample projects come with optimized artifacts, hence only the artifact
        # path is required.
        model_path = '/opt/awscam/artifacts/mxnet_deploy_ssd_resnet50_300_FP16_FUSED.xml'
        # Load the model onto the GPU.
        client.publish(topic=iot_topic, payload='Loading object detection model')
        model = awscam.Model(model_path, {'GPU': 1})
        client.publish(topic=iot_topic, payload='Object detection model loaded')
        # Set the threshold for detection
        detection_threshold = 0.2
        # The height and width of the training set images
        input_height = 300
        input_width = 300
        # Do inference until the lambda is killed.
        frame_number = 0
        while True:
            frame_number += 1
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Resize frame to the same size as the training set.
            frame_resize = cv2.resize(frame, (input_height, input_width))
            # Run the images through the inference engine and parse the results using
            # the parser API, note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a ssd model,
            # a simple API is provided.
            #
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(frame_resize))
            # Compute the scale in order to draw bounding boxes on the full resolution
            # image.
            yscale = float(frame.shape[0] / input_height)
            xscale = float(frame.shape[1] / input_width)
            s3 = boto3.resource('s3')

            ########
            BUCKET_NAME = 'shell-deeplens-images-and-coordinates'
            s3 = boto3.resource('s3')
            s3.Bucket(BUCKET_NAME).download_file('public/coordinates/1.json', '/tmp/1.json')
            with open('/tmp/1.json') as json_file:
                cord = json.load(json_file)
            ########
            #             cord = s3.Object('shell-deeplens-images-and-coordinates', 'public/coordinates/1.json')
            #             cord = cord.get()['Body'].read()
            #             cord = json.loads(str(cord)[2:-1].replace("'", "\""))

            #             with open('/tmp/1.json') as json_file:
            #                 cord = json.load(json_file)
            contours = [[cord['coordinates'][i]['x'], cord['coordinates'][i]['y']] for i in
                        range(len(cord['coordinates']))]
            pts = [[(xscale + 1) * cord['coordinates'][i]['x'], yscale * cord['coordinates'][i]['y']] for i in
                   range(len(cord['coordinates']))]

            contours = [np.array(contours, dtype=np.int32)]
            pts = np.array(pts, dtype=np.int32)
            # Dictionary to be filled with labels and probabilities for MQTT
            cloud_output = {}
            # Get the detected objects and probabilities
            for obj in parsed_inference_results[model_type]:
                # if output_map[obj['label']] == 'person' and obj['prob'] > detection_threshold:
                if obj['prob'] > detection_threshold:
                    if output_map[obj['label']] == 'person':
                        # Add bounding boxes to full resolution frame
                        #                         xmin = xs*xmin + (xmin-150)+ 150 = xmin(xs+1)
                        xmin = int(xscale * obj['xmin']) \
                               + int((obj['xmin'] - input_width / 2) + input_width / 2)
                        ymin = int(yscale * obj['ymin'])
                        xmax = int(xscale * obj['xmax']) \
                               + int((obj['xmax'] - input_width / 2) + input_width / 2)
                        ymax = int(yscale * obj['ymax'])
                        # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                        # for more information about the cv2.rectangle method.
                        # Method signature: image, point1, point2, color, and tickness.
                        #######################################################################
                        #                         contours = [np.array([[0,0],[0,150],[150,150],[150,0]], dtype=np.int32)]
                        lowerleft = cv2.pointPolygonTest(contours[0], (obj['xmin'], obj['ymax']), True)
                        lowerright = cv2.pointPolygonTest(contours[0], (obj['xmax'], obj['ymax']), True)
                        if lowerleft > 0 or lowerright > 0:
                            #######################################################################
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 10)
                            # Amount to offset the label/probability text above the bounding box.
                            text_offset = 15
                            # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                            # for more information about the cv2.putText method.
                            # Method signature: image, text, origin, font face, font scale, color,
                            # and tickness
                            speak.speak('Violation happened at Camera one location')
                            cv2.putText(frame, "{}: {:.2f}%".format(output_map[obj['label']],
                                                                    obj['prob'] * 100),
                                        (xmin, ymin - text_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 165, 20), 6)
                            # Store label and probability to send to cloud
                            cloud_output[output_map[obj['label']]] = obj['prob']
                            dic1 = {"action": "DETECTED_HUMAN",
                                    "cameraId": 1}
                            client.publish(topic=iot_topic, payload=json.dumps(dic1))
                            write_image_to_s3(frame_resize)
                            sns_client = boto3.client('sns')
                            file_name = 'Camera-1: ' + str(datetime.datetime.now())[:19] + '.jpg'
                            image_url = 'https://s3-us-east-1.amazonaws.com/shell-deeplens-images-and-coordinates/violations/' + file_name
                            notification_txt = 'Violation happened at Camera-1 location on: ' + str(
                                datetime.datetime.now())[
                                                                                                :19] + '. An image of the violation is available at: ' + image_url
                            resp = sns_client.publish(
                                TopicArn='arn:aws:sns:us-east-1:452950188950:shell',
                                Message=json.dumps(
                                    {
                                        "message": notification_txt
                                    }
                                )
                            )
            #             if frame_number == 1:
            if (float(frame_number / 10000)).is_integer():
                f1_url = write_frame1_to_s3(frame_resize)
                dic2 = {
                    "action": "PHOTO_FROM_DEEPLENS",
                    "cameraId": 1
                }
                client.publish(topic=iot_topic, payload=json.dumps(dic2))
            #             pts = np.array([[0,0],[0,yscale *150],[(xscale+1) *150,yscale *150],[(xscale+1) *150,0]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 0, 255), 10)
            #             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 10)

            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
            # Send results to the cloud
            client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in object detection lambda: {}'.format(ex))


greengrass_infinite_infer_run()