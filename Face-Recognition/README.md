
# Facial Recognition

### Facial recognition a well known term among tech enthusiast, It has been widely used as a tool to recognise persons and to differentiate them with respect to their facial attributes.

- Tested on python-3
- Package required:
    1. dlib,opencv
    2. sklearn,pandas_ml
    3. pandas, numpy
    4. tmdbsimple, pytvdbapi

- import API


```python
from Face_recog_API import *
```

- Data collection 


```python
tv_series_name = 'True Detective'

face_rec = FacialRecognition(data_path=tv_series_name,face_detector_type='hog')

cast_list = face_rec.get_cast_name_tvdb(tv_series_name)
print(cast_list)
```

    loading models
    ['Matthew McConaughey', 'Woody Harrelson', 'Rachel McAdams', 'Kelly Reilly', 'Vince Vaughn', 'Taylor Kitsch', 'Colin Farrell', 'Michelle Monaghan', 'Kevin Dunn', 'Michael Potts', 'Tory Kittles', 'Alexandra Daddario']


- Data collection


```python
face_rec.downloaded_images(cast_list)
```

- Data Preparation


```python
[X,y]=face_rec.prepare_data()
```

- Training Model


```python
face_rec.partial_train_model(X,y,minm_image_process=30,threshold_accuracy=.9,classes=range(1,len(cast_list)+1))
```

    entering training module
    StratifiedShuffleSplit(n_splits=10, random_state=0, test_size=0.2,
                train_size=None)
    inside preprocessing function
    returning from preprocess data
    classes must be passed on the first call to partial_fit.
    accuracy in iteration  1  is = 0.910958904109589
    returning from train module


    /home/prateek/.virtualenvs/cv3/lib/python3.5/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
      DeprecationWarning)
    /home/prateek/.virtualenvs/cv3/lib/python3.5/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
      DeprecationWarning)


- Viewing output


```python
cap = cv2.VideoCapture('True Detective Season 1 - Woody Harrelson & Matthew McConaughey\'s Fight Scene (HBO) --VvkSTRVaOCs.mp4')
while cap.isOpened():
    Disp.clear_output(wait=True)
    ret,frame = cap.read()
    op = face_rec.get_pred_on_frame(frame.copy())
    cv2.imshow("preview",op)
    k = cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
cap.release()
    
```
