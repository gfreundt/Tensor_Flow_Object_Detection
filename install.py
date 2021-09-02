import sys, os, time, uuid, subprocess, shutil
from pathlib import Path

if sys.argv[1] == "0":

    # 0. Check if Running on Windows
    if os.name != 'nt':
        raise Exception('This script is only valid for Windows users.')

    # 1. 
    ROOT_PATH = os.getcwd() # Should be path that finished in "Tensor_Flow_Object_Detection"
    COURSE_PATH = os.path.join(ROOT_PATH, 'TFODCourse')

    print(f'\n\nProcess will begin installing in {ROOT_PATH}\n')
    _ = input('PRESS ENTER TO ACCEPT AND CONTINUE')

    #2. Load repositories, install dependencies in a virtual environment

    # Clone Nick's Github repository and switch to that folder
    NICK_REPOSITORY = 'https://github.com/nicknochnack/TFODCourse'
    os.chdir(ROOT_PATH)
    os.system(f'git clone {NICK_REPOSITORY}')
    os.chdir(COURSE_PATH)

    # Create and activate virtual environment ("tfod") and install python dependencies in "requirements.txt
    os.system(r'python -m venv tfod && .\tfod\Scripts\activate && pip install -r ..\requirements.txt')

    # Give back control to batch script to run next part inside virtual environment

if sys.argv[1] == "1":

    import cv2

    ROOT_PATH = os.getcwd() # Should be path that finished in "Tensor_Flow_Object_Detection"
    COURSE_PATH = os.path.join(ROOT_PATH, 'TFODCourse')

    # 3. Load Labels text file and Number of Images to Collect for each label
    os.chdir(ROOT_PATH)
    with open('labels.txt', mode='r') as file:
        labels = [i.strip() for i in file.readlines()]
    print(f"Will save images into the following folders: {labels} \n")
    number_imgs = input('Images per label (default = 5):')
    if not number_imgs.isdigit():
        number_imgs = 5
    elif int(number_imgs) < 1:
        number_imgs = 5
    else:
        number_imgs = int(number_imgs)

    #4. Setup Folders
    IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')
    os.chdir(COURSE_PATH)
    if not os.path.exists(IMAGES_PATH):
        for label in labels:
            path = os.path.join(IMAGES_PATH, label)
            Path(path).mkdir(parents=True, exist_ok=True)

    #5. Capture Images
    for label in labels:
        cap = cv2.VideoCapture(0)
        print(f'Collecting images for {label}')
        time.sleep(5)
        for imgnum in range(number_imgs):
            print(f'Collecting image {imgnum}')
            ret, frame = cap.read()
            imgname = os.path.join(IMAGES_PATH,label,label+'@'+f'{str(uuid.uuid1())}.jpg')
            cv2.imwrite(imgname, frame)
            cv2.imshow('frame', frame)
            time.sleep(2)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

    #6. Clone Image Labelling Script
    LABELIMG_PATH = os.path.join(COURSE_PATH, 'Tensorflow', 'labelImg')
    if not os.path.exists(LABELIMG_PATH):
        Path(LABELIMG_PATH).mkdir(parents=True, exist_ok=True)
        os.system(f"git clone https://github.com/tzutalin/labelImg {LABELIMG_PATH}")
        os.system(f"cd {LABELIMG_PATH} && pyrcc5 -o libs/resources.py resources.qrc")

    # Give back control to batch script to execute labeling script

    
if sys.argv[1] == "2":

    import random

    #7. Select images randomly for training and testing in new folder tree
    ROOT_PATH = os.getcwd() # Should be path that finished in "Tensor_Flow_Object_Detection"
    COURSE_PATH = os.path.join(ROOT_PATH, 'TFODCourse')
    IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images')
    COL_IMAGES_PATH = os.path.join(IMAGES_PATH, 'collectedimages')
    TESTING_IMAGES_PATH = os.path.join(IMAGES_PATH,'testing')
    LEARNING_IMAGES_PATH = os.path.join(IMAGES_PATH,'learning')

    os.chdir(COURSE_PATH)
    # create testing and learning folders to store collected images and their corresponding labels
    Path(TESTING_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    Path(LEARNING_IMAGES_PATH).mkdir(parents=True, exist_ok=True)

    SPLIT_FOR_TESTING = 0.5
    for label in os.listdir(COL_IMAGES_PATH):
        path = os.path.join(COL_IMAGES_PATH, label)
        # creates list with names (no extensions) in label folder
        files = [i.split('.')[0] for i in os.listdir(path) if 'jpg' in i]
        # splits list into test and learn groups randomly
        test_files = random.sample(files,int(len(files)*SPLIT_FOR_TESTING))
        learn_files = [i for i in files if i not in test_files]
        print('test', test_files)
        print('learn', learn_files)

        # moves image and label frome each group into corresponding folder
        _ = [[shutil.move(os.path.join(path, i+j), TESTING_IMAGES_PATH) for i in test_files] for j in ('.jpg', '.xml')]
        _ = [[shutil.move(os.path.join(path, i+j), LEARNING_IMAGES_PATH) for i in learn_files] for j in ('.jpg', '.xml')]

        # erase empty folder
        shutil.rmtree(path)

if sys.argv[1] == "3":

    import wget
    
    #8. Define URIs
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'

    paths={
        'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow','models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
        'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
        'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
        'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
        'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
        'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
        'PROTOC_PATH':os.path.join('Tensorflow','protoc')
    }

    files = {
        'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }

    #9. Create folder tree
    ROOT_PATH = os.getcwd() # Should be path that finished in "Tensor_Flow_Object_Detection"
    COURSE_PATH = os.path.join(ROOT_PATH, 'TFODCourse')
    os.chdir(COURSE_PATH)

    for path in paths.values():
        if not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)

    # https://www.tensorflow.org/install/source_windows


    #10. Clone Tensorflow Model
    if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
        os.system(f'git clone https://github.com/tensorflow/models {paths["APIMODEL_PATH"]}')

    #11. Install Tensorflow Object Detection
    url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url)
    shutil.move('protoc-3.15.6-win64.zip', paths['PROTOC_PATH'])
    
    os.system(f"cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip")
    os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   
    os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\\\packages\\\\tf2\\\\setup.py setup.py && python setup.py build && python setup.py install")
    
    #12. Activate virtual environment, execute pip install -e and upgrade tensorflow module
    os.system(r'.\Scripts\activate && cd Tensorflow/models/research/slim && pip install -e . && pip install tensorflow --upgrade')

    #13. Run Verification 
    
    VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    SCRIPT = [r'.\tfod\Scripts\activate', '&&', 'python', VERIFICATION_SCRIPT]
    print('Running verification script')
    response = subprocess.run(SCRIPT, text=True, capture_output=True, shell=True)
    response = response.stderr.strip()
    if "OK" in response[-30:]:
        print('Verification Script OK. Process Complete.')
    else:
        raise Exception ('Did not pass Verification') 