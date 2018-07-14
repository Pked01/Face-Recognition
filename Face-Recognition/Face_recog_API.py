
import os,sys,re,time,dlib,subprocess,pickle,cv2
import tmdbsimple as tmdb
from pytvdbapi import api
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import pandas_ml as pdml
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import Iterable
from sklearn.model_selection import StratifiedShuffleSplit

import IPython.display as Disp


class label_encoder():
    """
    label_dict: prepared dictionary can be used
    data_path:saving loading path of file
    """
    def __init__(self,labels_dict=None,data_path='models/labels.pickle'):
        if data_path is not None:
            self.data_path=data_path
            if labels_dict is None:
                if os.path.exists(self.data_path) :  
                    self.labels=pickle.load(open(self.data_path,'rb'))
                else:
                    try:
                        os.mkdir(os.path.dirname(self.data_path))
                    except:
                        pass
                    self.labels={}
            else:
                self.labels=labels_dict
        else : 
            raise ValueError('Give a valid dumping path for labels.pickle')

    def fit(self,x):
        """
        x:list or a single element which has to be encoded
        """
        if ((isinstance(x, Iterable)) & (type(x)!=str)):
            iter1=list(set(x)-set(self.labels.keys()))
            for i in iter1:
                self.labels[i]=len(self.labels.keys())+1
        else:
            if x not in self.labels.keys():
                self.labels[x]=len(self.labels.keys())+1 
    def transform(self,key):
        """
        key: set(list/tuple) of elements for which values has to be retrieved 
        """
        l=[]
        if ((isinstance(key, Iterable))&(type(key)!=str)):
            print("its an iterable")
            for i in key:
                try:
                    l.append(self.labels[i])
                except Exception as e:
                    print("iterable error",e)
            return l
        else:
            try:
                return self.labels[key]
            except Exception as e:
                print("error",e)
    def save(self):
        try:
            pickle.dump(self.labels,open(self.data_path,'wb'),protocol=2)
        except:
            #os.mkdir(self.data_path)
            pickle.dump(self.labels,open(self.data_path,'wb'),protocol=2)
    #--------------------------------------------------------------------------------------------------------------






class FacialRecognition(object):

    
    def __init__(self,data_path,face_detector_type='cnn'):
        """
        face_detector_type= hog(cpu)/cnn(gpu)
        """
        self.data_path = data_path
        self.face_detector_type = face_detector_type
        self.cwd = os.getcwd()
        self.sys_path = os.path.expanduser('~')
        self.model_files_path = os.path.join(self.sys_path,'dlib_model_files')
        self.dump_file_path = os.path.join(self.cwd,self.data_path,'models')
        os.makedirs(self.dump_file_path, exist_ok=True)
        try :
            self.labels = pickle.load(open(os.path.join(self.dump_file_path,'labels.pickle'),'rb'))
        except:
            self.labels = None
        try :
            self.base_model = pickle.load(open(os.path.join(self.dump_file_path,'sgd_model_resampled.pickle'),'rb'))
        except:
            self.base_model = None
        
        if ((os.path.exists(self.model_files_path)) and (len(os.listdir(self.model_files_path))>3)) :
            # load all models 
            self.__load_models()
            

        else:
            os.makedirs(self.model_files_path,exist_ok=True)
            #cnn
            
            mmod_face_link = " http://dlib.net/files/mmod_human_face_detector.dat.bz2"
            print("downloading-->"+mmod_face_link)
            subprocess.call(("wget -P "+self.model_files_path+mmod_face_link).split(" "))
            #wget.download(mmod_face_link,self.model_files_path,bar=wget.bar_thermometer)
            self.__uncompress_file(os.path.join(self.model_files_path,"mmod_human_face_detector.dat.bz2"))
            
            #68 face landmarks
            face_landmark_link = " http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            print("downloading-->"+face_landmark_link)
            #wget.download(face_landmark_link,self.model_files_path,bar=wget.bar_thermometer)
            subprocess.call(("wget -P "+self.model_files_path+face_landmark_link).split(" "))
            self.__uncompress_file(os.path.join(self.model_files_path,"shape_predictor_68_face_landmarks.dat.bz2"))
            
            #resnet model
            resnet_link = " http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
            print("downloading-->"+resnet_link)
            #wget.download(resnet_link,self.model_files_path,bar=wget.bar_thermometer)
            subprocess.call(("wget -P "+self.model_files_path+resnet_link).split(" "))
            self.__uncompress_file(os.path.join(self.model_files_path,"dlib_face_recognition_resnet_model_v1.dat.bz2"))
            # path for saving all intermediate models 
            self.__load_models()
        


#         self.face_detector = dlib.get_frontal_face_detector()
#         self.shape_pred = dlib.shape_predictor('/dlib_model_files/shape_predictor_68_face_landmarks.dat')
#         self.facerec = dlib.face_recognition_model_v1('/dlib_model_files/dlib_face_recognition_resnet_model_v1.dat')
    def __load_models(self):
        print("loading models")
        if self.face_detector_type=='cnn':
            self.face_detector = dlib.cnn_face_detection_model_v1(os.path.join(self.model_files_path,'mmod_human_face_detector.dat'))
        else:
            self.face_detector = dlib.get_frontal_face_detector()
        self.shape_pred = dlib.shape_predictor(os.path.join(self.model_files_path,'shape_predictor_68_face_landmarks.dat'))
        self.facerec = dlib.face_recognition_model_v1(os.path.join(self.model_files_path,'dlib_face_recognition_resnet_model_v1.dat'))

    def __uncompress_file(self,readpath,writepath=None):
        if writepath is None:
            writepath = os.path.join(os.path.dirname(readpath) , os.path.basename(readpath).replace('.bz2',''))
        zipfile = bz2.BZ2File(readpath) # open the file
        data = zipfile.read() # get the decompressed data
        newfilepath = readpath[:-4] # assuming the filepath ends with .bz2
        open(writepath, 'wb').write(data) # write a uncompressed file

    def detect_face(self,image,return_image=False,upsample_num_times=0):
        rects,scores,weights= self.face_detector.run(image,upsample_num_times=upsample_num_times)
        op = []
        for idx,r in enumerate(rects):
            im_c = image.copy()
            op.append([(r.left(),r.top()),(r.right(),r.bottom())])
            im_c = cv2.rectangle(im_c,op[idx][0],op[idx][1],(0,255,0))
        if return_image:
            return op,scores,weights,im_c
        return op,scores,weights




    def get_face_landmark(self,image,return_image = False):
        dets = self.shape_pred(image,self.face_detector(image)[0])
        tot_landmark = dets.parts()
        if return_image:
            im_c = image.copy()
            for i in tot_landmark:
                cv2.circle(im_c,(i.x,i.y),1,(0,255,0),1)
            return tot_landmark,im_c
        return tot_landmark


    #face_descriptor = facerec.compute_face_descriptor(im, shape_pred(im,f_detector(im)[0]))

    # It should also be noted that you can also call this function like this:
    #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100)
    # The version of the call without the 100 gets 99.13% accuracy on LFW
    # while the version with 100 gets 99.38%.  However, the 100 makes the
    # call 100x slower to execute, so choose whatever version you like.  To
    # explain a little, the 3rd argument tells the code how many times to
    # jitter/resample the image.  When you set it to 100 it executes the
    # face descriptor extraction 100 times on slightly modified versions of
    # the face and returns the average result.  You could also pick a more
    # middle value, such as 10, which is only 10x slower but still gets an
    # LFW accuracy of 99.3%.

    def get_face_embedding(self,image):
        """
        return embedding of an face image 
        """
        if self.face_detector_type=='cnn':
            face_descriptor = [self.facerec.compute_face_descriptor(image, self.shape_pred(image,i.rect))  for i in self.face_detector(image,1)]
        else:
            face_descriptor = [self.facerec.compute_face_descriptor(image, self.shape_pred(image,i))  for i in self.face_detector(image)]
            
        return face_descriptor
    
    def get_cast_name_tmdb(self,series_name='two and half man'):
        """
        tv series or title of movie
        returns list of cast
        """
        tmdb.API_KEY = 'd8eb79cd5498fd8d375ac1589bfc78ee'
        search = tmdb.Search()
        response = search.tv(query=series_name)
        tv1=tmdb.TV(id=response['results'][0]['id'])
        return [i['name'] for i in tv1.credits()['cast']]


    def get_cast_name_tvdb(self,series_name='two and half man'):
        """
        tv series name 
        return cast list
        """
        db = api.TVDB("05669A6CC3005169", actors=True, banners=True)
        result = db.search(series_name, "en")
        show = result[0]
        show.update()
        return show.Actors
    ###---------------------- download utility-------------------------------------------------------------------------
    #Downloading entire Web Document (Raw Page Content)
    #
    def download_page(self,url):
        version = (3,0)
        cur_version = sys.version_info
        if cur_version >= version:     #If the Current Version of Python is 3.0 or above
            import urllib.request    #urllib library for Extracting web pages
            try:
                headers = {}
                headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
                req = urllib.request.Request(url, headers = headers)
                resp = urllib.request.urlopen(req)
                respData = str(resp.read())
                return respData
            except Exception as e:
                print(str(e))
        else:                        #If the Current Version of Python is 2.x
            #import urllib2
            try:
                headers = {}
                headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
                req = urllib.Request(url, headers = headers)
                response = urllib.request.urlopen(req)
                page = response.read()
                return page
            except:
                return"Page Not found"


    #Finding 'Next Image' from the given raw page
    def _images_get_next_item(self,s):
        start_line = s.find('rg_di')
        if start_line == -1:    #If no links are found then give an error!
            end_quote = 0
            link = "no_links"
            return link, end_quote
        else:
            start_line = s.find('"class="rg_meta"')
            start_content = s.find('"ou"',start_line+1)
            end_content = s.find(',"ow"',start_content+1)
            content_raw = str(s[start_content+6:end_content-1])
            return content_raw, end_content


    #Getting all links with the help of '_images_get_next_image'

    def _images_get_all_items(self,page):
        items = []
        while True:
            item, end_content = self._images_get_next_item(page)
            if item == "no_links":
                break
            else:
                items.append(item)      #Append all the links in the list named 'Links'
                time.sleep(0.1)        #Timer could be used to slow down the request for image downloads
                page = page[end_content:]
        return items


    def downloaded_images(self,cast_list,data_path=None,series_name=""):
        """
        data_path:  save path for data
        """
        if data_path is None:
            data_path = self.data_path
        if not os.path.exists(data_path):
            os.makedirs(data_path,exist_ok=True)

        for cast in cast_list:
            #print(cast)
            os.makedirs(os.path.join(data_path,cast),exist_ok=True)
            files_folder = os.listdir(os.path.join(data_path,cast))
            if len(files_folder)>0:
                op = input("Folder contains files --"+cast+"--  you want to continue(y/n)")
                if op=='y':
                    pass
                if op=='n':
                    continue
                
            
            ########### Edit From Here ###########

            #This list is used to search keywords. You can edit this list to search for google images of your choice. You can simply add and remove elements of the list.
            cast=re.sub(r'[^\x00-\x7F]+',' ', cast)
            search_keyword = [cast+" "+series_name]

            #This list is used to further add suffix to your search term. Each element of the list will help you download 100 images. First element is blank which denotes that no suffix is added to the search keyword of the above list. You can edit the list by adding/deleting elements from it.So if the first element of the search_keyword is 'Australia' and the second element of keywords is 'high resolution', then it will search for 'Australia High Resolution'
            keywords = ['']

            ########### End of Editing ###########

            ############## Main Program ############
            t0 = time.time()   #start the timer

            #Download Image Links
            i= 0
            while i<len(search_keyword):
                items = []
                iteration = "Item no.: " + str(i+1) + " -->" + " Item name = " + str(search_keyword[i])
                print (iteration)
                print ("Evaluating...for", cast)
                search_keywords = search_keyword[i]
                search = search_keywords.replace(' ','%20')

#                  #make a search keyword  directory
#                 try:
#                     os.makedirs(os.path.join(data_path,cast))
#                 except Exception as e:
#                     if e.errno != 17:
#                         raise   
#                     # time.sleep might help here
#                     pass

                j = 0
                while j<len(keywords):
                    pure_keyword = keywords[j].replace(' ','%20')
                    url = 'https://www.google.com/search?q=' + search + pure_keyword + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
                    raw_html =  (self.download_page(url))
                    time.sleep(0.1)
                    items = items + (self._images_get_all_items(raw_html))
                    j = j + 1
                #print ("Image Links = "+str(items))
                print ("Total Image Links = "+str(len(items)))
                print ("\n")


                #This allows you to write all the links into a test file. This text file will be created in the same directory as your code. You can comment out the below 3 lines to stop writing the output to the text file.
                info = open(data_path+'output.txt', 'a')        #Open the text file called database.txt
                info.write(str(i) + ': ' + str(search_keyword[i-1]) + ": " + str(items) + "\n\n\n")         #Write the title of the page
                info.close()                            #Close the file

                t1 = time.time()    #stop the timer
                total_time = t1-t0   #Calculating the total time required to crawl, find and download all the links of 60,000 images
                print("Total time taken: "+str(total_time)+" Seconds")
                print ("Starting Download...")

                ## To save imges to the same directory
                # IN this saving process we are just skipping the URL if there is any error

                k=0
                errorCount=0
                while(k<len(items)):
                    from urllib import request
                    #from urllib import URLError, HTTPError

                    try:
                        req = request.Request(items[k], headers={"User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
                        response = request.urlopen(req,None,15)
                        if os.path.exists(data_path):
                            output_file = open(os.path.join(data_path,cast,str(cast)+'_'+str(k+1)+".jpg"),'wb')

                        data = response.read()
                        output_file.write(data)
                        response.close();

                        print("completed ====> "+str(k+1))

                        k=k+1;

                    except IOError:   #If there is any IOError

                        errorCount+=1
                        print("IOError on image "+str(k+1))
                        k=k+1;

                    except request.HTTPError as e:  #If there is any HTTPError

                        errorCount+=1
                        print("HTTPError"+str(k))
                        k=k+1;
                    except request.URLError as e:

                        errorCount+=1
                        print("URLError "+str(k))
                        k=k+1;
                    except Exception as e:
                        print(e)



                i = i+1

            print("\n")
            print("Everything downloaded!")
            print("\n"+str(errorCount)+" ----> total Errors")
            Disp.clear_output()

            #----End of the main program ----#

    #--------------------------------------------------------------------------------------------------------
            # In[ ]:
    def get_stratified_sample(self,X,y,verbose=True,test_size=.2):
        sss = StratifiedShuffleSplit(n_splits=10, test_size=test_size, random_state=0)
        sss.get_n_splits(X, y)
        print(sss)       
        for train_index, test_index in sss.split(X, y):
            if verbose:
                print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        return [X_train,X_test,y_train,y_test]

    #get_name_only=lambda name:''.join([i for i in name if not i.isdigit()]).replace("_","").lower()
    get_name_only=lambda self, name:re.sub('[^A-Za-z]+', '', name).lower()
    get_name_only.__name__='get_name_only'
    
    inv_map=lambda self,my_map: {v: k for k, v in my_map.items()}
    inv_map.__name__='inverse_mapping'
    def get_image_files(self,directory_path=None,return_only_paths=True):
        """
        directory_path=path of parent directory
        """
        print("inside get_image_files function")
        if directory_path is None:
            directory_path = self.data_path
        paths={}
        image_files={}
        for root, dirs, files in os.walk(directory_path, topdown=False):
            for name in files:
                path=os.path.join(root, name)
                if name.endswith('.jpg'):
                    #removing name,number_from_data
                    file_name=name.replace(".jpg","")#root.split('/').pop()+name.replace(".jpg","")  
                    #if file_name in file_name_filt:
                    print(path,file_name)
                    try:
                        if return_only_paths:
                            paths[file_name]=path
                        else:
                            image_files=cv2.imread(path)
                    except:
                        print("encoding error")

            Disp.clear_output()
        print("returning from get_image_files function")    
        if return_only_paths:
            return paths
        else:
            return image_files    
    def get_label_charac_dict(self,directory_path=None):
        """
        It loads root directory and get their characters name and assign labels to them

        """
        print("inside get_label_charac_dict function")
        if directory_path is None:
            directory_path = self.data_path
        try:
            charac_names=pickle.load(open(os.path.join(directory_path,'charac_names.pickle'),'rb'))
        except:
            charac_names={}

        im_files = self.get_image_files(directory_path)
        for file_name in im_files.keys():
            charac_names[file_name]=self.get_name_only(file_name)
        lbl_enc = label_encoder(data_path=os.path.join(directory_path,'models','labels.pickle'))
        lbl_enc.fit(charac_names.values())
        lbl_enc.save()
        labels=lbl_enc.labels
        pickle.dump(charac_names,open(os.path.join(directory_path,'models','charac_names.pickle'),'wb'),protocol=2)
        print("returning from get_label_charac_dict function")

        return {'charac_names':charac_names,'labels':labels}
    def prepare_data(self,directory_path=None,l_threshold=20,r_threshold=None,load_data=False):
        """
        returns a vector X(128 sized) and encoded label y
        directory_path=source path of images folder, with each image have parent folder as label of it
        l_threshold: minimum images for a profile in images list
        r_threshold: maximum images for a profile in images list
        Autoresizing is done for any image more than 300 in max dimension
        minm_num: minimum number of images in a class
        output:get_all_files from folders and get encoding out of it
        dump_file_path: dumps a tuple of X,y
        """
        if load_data:
            try: 
                [X,y] = pickle.load(open(os.path.join(self.dump_file_path,'[X,y]_encoded_file.pickle'),'rb'))
            except Exception as e:
                print(e)
        else:
            print("entering into prepare data")
            if directory_path is None:
                directory_path = self.data_path
            jsn=self.get_label_charac_dict(directory_path)
            charac_names = jsn['charac_names']
            self.labels = jsn['labels']
            encoding_files = {}
            t=pd.DataFrame(list(charac_names.values()))[0].value_counts()
            t=t[t>=l_threshold]
            if r_threshold is not None:
                t=t[t<=r_threshold]
            t1=t.index
            file_name_filt=[]
            print('total unique matches with criteria',t1.shape)
            for k,v in charac_names.items():
                if v in t1:
                    file_name_filt.append(k)
            del t1

        #     for root, dirs, files in os.walk(directory_path, topdown=False):
        #         for name in files:
        #             path=os.path.join(root, name)
        #             if name.endswith('.jpg'):
        #                 #removing name,number_from_data
        #                 file_name=root.split('/').pop()+name.replace(".jpg","")  
        #                 if file_name in file_name_filt:
        #                     print(path,file_name)
            im_files=self.get_image_files(directory_path)
            for file_name,path in im_files.items():
                try:
                    image = cv2.imread(path)
                    image_res = image.copy()
                    max_dim = max(image.shape[0:2])
                    if max_dim > 300.0:
                        #image_res=scipy.misc.imresize(image,300.0/max(image.shape[0:2]))
                        image_res = cv2.resize(image_res,(0,0), fx=300.0/max_dim, fy=300.0/max_dim)
                    encoding_1 = self.get_face_embedding(image_res)
                    if len(encoding_1)==1:
                        encoding_files[file_name] = list(encoding_1[0])

                except Exception as e:
                    print("encoding error",e)
                Disp.clear_output()
                        #charac_names[file_name]=charac_name
        #     if dump_file_path is not None:
        #         print('dumping output')
        #         pickle.dump(encoding_files,\
        #                     open(dump_file_path+'_encoded_file.pickle','wb'),protocol=2)
    #         l=list(encoding_files.keys())
    #         for k in l:
    #             if len(encoding_files[k])!=1:
    #                 del encoding_files[k]
    #             else:
    #                 encoding_files[k]=encoding_files[k][0]
            encoding_df=pd.DataFrame(encoding_files).T
            encoding_df['label_enc']=[self.labels[self.get_name_only(i)] for i in encoding_df.index]
            X=encoding_df.iloc[:,:128].values
            y=encoding_df['label_enc'].values
            if self.dump_file_path is None:
                self.dump_file_path = os.path.join(directory_path,"models")
            else:
                os.makedirs(self.dump_file_path,exist_ok=True)
                print('dumping output')
                pickle.dump([X,y],open(os.path.join(self.dump_file_path,'[X,y]_encoded_file.pickle'),'wb'),protocol=2)
            print("returning prepare data")
            return [X,y]
    
    def process_data(self,X,y,minm_num=30):
        """
        SMOTE and resampling
        """
        print("inside preprocessing function")
        df = pdml.ModelFrame(X,y)
        sampler=df.imbalance.over_sampling.SMOTE()
        sampled=df.fit_sample(sampler)
        total_classes=len(np.unique(y))
        if sampled.shape[0]/total_classes<minm_num:
            resampled_class=resample(sampled.iloc[:,1:].values,sampled.target.values,n_samples=2*minm_num*total_classes)
            sampled=pd.DataFrame(resampled_class[0])
            sampled['.target']=resampled_class[1]

        desampled=sampled.groupby('.target').apply(lambda x: pd.DataFrame(x).sample(n=minm_num))
        desampled.reset_index(drop=True,inplace=True)
        print("returning from preprocess data")
        return [desampled[list(range(128))],desampled['.target']]

    def partial_train_model(self,X,y,minm_image_process=None,threshold_accuracy=.9,classes=range(20)):
        """
        incremental training module(SGD)
        returns a new model after partial fit on give data
        X=128 sized vector 
        y=labels of vectors
        minm_image_process='how many images of a specific label have to be trained, oversampling undersampling is done,  
        classes:number of that is going to be used in this model have to defined in advance
        """
        print("entering training module")
        self.base_model=SGDClassifier(loss='log',n_jobs=7,\
                      shuffle=True,class_weight=None,warm_start=False\
                      ,n_iter = np.ceil(10**6 / 600),average=True)
        
        [X_train,X_test,y_train,y_test]=self.get_stratified_sample(X,y,verbose=False)
        if minm_image_process is not None:
            [X_processed,y_processed]=self.process_data(X_train,y_train,minm_num=minm_image_process)
        else:
            [X_processed,y_processed]=[X_train,y_train]
 
        accuracy=0
        idx=0
        while accuracy<threshold_accuracy:
            try:
                self.base_model.partial_fit(X_processed,y_processed)
            except Exception as e:
                print(e)
                self.base_model.partial_fit(X_processed,y_processed,classes=classes)
            y_pred=self.base_model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            print("accuracy in iteration ",idx+1,' is =',accuracy)
            idx+=1
            if idx>10:
                break
        if self.dump_file_path is  None:
            self.dump_file_path = os.path.join(self.data_path,'models')
        else:
            os.makedirs(self.dump_file_path,exist_ok=True)
            pickle.dump(self.base_model,open(os.path.join(self.dump_file_path,'sgd_model_resampled.pickle'),'wb'))
        print("returning from train module") 
        
    def get_pred_on_frame(self,frame,verbose=False):
        """
        provide prediction on a frame 
        model: classifier model
        data_path: loading relevent file from the source 
        """
        inv_labels=self.inv_map(self.labels)
        #frame=frame.mean(axis=2)
        face_locations = self.face_detector(frame)
        if len(face_locations)>0:
            face_encodings = self.get_face_embedding(frame)
            if verbose:
                print("number of faces detected",len(face_locations))
            face_names = []
            for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
                try:
                    #print(face_encoding)
                    match = self.base_model.predict(np.array(face_encoding).reshape([1,128]))[0]
                    predict_probab=self.base_model.predict_proba(np.array(face_encoding).reshape([1,128]))[0].max()
                    #bin_prob=math.exp(predict_probab[match])/sum([math.exp(i)for i in predict_probab])
                    #bin_prob=(predict_probab[match]-np.mean(predict_probab))/np.std(predict_probab)
                    if verbose:
                        print(match,inv_labels[match],predict_probab)
                    face_names.append(inv_labels[match]+ '(p='+str(np.round(predict_probab,3))+')')
                    #face_names.append(inv_labels[match]+ ' prediction probability='+str(1/(1+math.exp(-bin_prob))))
                except Exception as e:
                    print(e)
            face_locations_final =[]
            if self.face_detector_type=='cnn':
                for face_location in face_locations:
                    face_locations_final.append([face_location.rect.top(),face_location.rect.right(),face_location.rect.bottom(),face_location.rect.left()])
            else:
                for face_location in face_locations:
                    face_locations_final.append([face_location.top(),face_location.right(),face_location.bottom(),face_location.left()])
            face_locations = face_locations_final



            # Label the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if not name:
                    continue

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        return frame

