import cv2, sys, numpy, os, time, datetime
from attendance_log import log_attendance

size = 2
classifier = 'haarcascade_frontalface_default.xml'
image_dir = 'images'
print("Face Recognition Starting ...")

(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(image_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(image_dir, subdir)
        for filename in os.listdir(subjectpath):
            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in ['.png','.jpg','.jpeg','.gif','.pgm']):
                continue
            path = os.path.join(subjectpath, filename)
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(im_width, im_height) = (112, 92)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

haar_cascade = cv2.CascadeClassifier(classifier)
webcam = cv2.VideoCapture(0)

def recognize_and_log():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    logged_faces = set()
    
    while True:
        rval, frame = webcam.read()
        if not rval:
            print("Gagal membuka webcam. Mencoba lagi...")
            continue
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        faces = haar_cascade.detectMultiScale(mini, scaleFactor=1.1, minNeighbors=5)
        for face_i in faces:
            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))

            prediction = model.predict(face_resize)
            name = names[prediction[0]] if prediction[1] < 90 else "Unknown"
            
            if name != "Unknown" and name not in logged_faces:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                photo_path = f"{name}_{today}_{timestamp.replace(':', '-')}.png"
                cv2.imwrite(photo_path, face_resize)
                log_attendance(name, today, timestamp, photo_path)
                logged_faces.add(name)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} - {prediction[1]:.0f}", (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_and_log()