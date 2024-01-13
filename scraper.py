from selenium import webdriver
import urllib.request
import time
import os
import numpy as np
import pandas as pd
import threading
from bs4 import BeautifulSoup


def download_image(title_name, src, alt, seq, our_dir):
    try:
        iamge_name = title_name + '_' + str(seq) +  '.jpg' # i.e: "JohnTravolta0.png"
        text_name = title_name + '_' + str(seq) +  '.txt' # i.e: "JohnTravolta0.png"
        image_path = os.path.abspath(os.path.join(os.getcwd(), our_dir, iamge_name)) # /home/user/Desktop/dirname
        text_path = os.path.abspath(os.path.join(os.getcwd(), our_dir, text_name)) # /home/user/Desktop/dirname
        if not os.path.exists(image_path):          
            urllib.request.urlretrieve(src, image_path) # download image
            with open(text_path, 'w', encoding='utf8') as fp:
                fp.write(alt)
    except Exception:
        return False
    return True


def browse_page(driver, title_name, sum_nums, our_dir, current_index, name):
    total_pages = 0
    current_page = int(current_index // 60 + 1)
    seq = int(current_index // 60) * 60 #initialize the file number.
    for i in range(sum_nums*10): # Loop for the number of pages you want to scrape.
        try:
#            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);') # Scroll to the end of page.
            time.sleep(2) # Wait for all the images to load correctly.
#            threading.sleep()
#            images = driver.find_elements_by_xpath("//img[contains(@class, 'gallery-asset__thumb gallery-mosaic-asset__thumb')]") # Find all images.
            all_datas = driver.find_elements_by_xpath("//div[@class='gallery-asset-schema' and @itemprop='image']") # Find all images. and @itemprop='image'
            if total_pages == 0:
                total_pages = driver.find_element_by_class_name("search-pagination__last-page").text
                print('total_pages:', total_pages)
        except:
            print('load data err')
            continue
        
        
        for da in all_datas: # For each image in one page:
            if seq < current_index:
                seq += 1
                continue
            flag = True
            try:       
                metas = da.get_attribute("innerHTML")
                soup = BeautifulSoup(metas)
                alt = soup.find_all('meta', itemprop='caption')[0]['content']
                src = soup.find_all('meta', itemprop='thumbnailUrl')[0]['content']
                lenth = len(alt.strip().split())
                if lenth < 5:
                    continue
                flag = download_image(title_name, src, alt, seq, our_dir) # And download it to directory
            except:
                print('download failed')
                flag = False
            if flag is True:
                seq += 1
            if seq % 4 == int(name) % 4:
                print_info[int(name) % 4] = '{} : {}'.format(title_name, seq)
                print("\r", "    ".join(print_info), end = "",flush=True)
            if seq >= sum_nums:
                break
        if seq >= sum_nums:
            break
        if current_page > int(total_pages):
            break
        else:
            current_page+=1
            
        while True:
            time.sleep(2)
            try:
                next_page = 'https://www.gettyimages.com/photos/{0}?family=editorial&page={1}&phrase={2}&sort=newest#license'.format(title_name, current_page, title_name)
                driver.get(next_page)
                break
#              nextpage = driver.find_element_by_css_selector('.search-pagination__button-icon--next').click() # Move to next page             
            except:
                print("no next page, try again:", title_name, '\n')
                continue
        time.sleep(2)
  

class myThread (threading.Thread):
    def __init__(self, name, data):
        threading.Thread.__init__(self)
        self.name = name
        self.data = data
    def run(self):
        print ("开始线程：" + self.name)
        #        driver = webdriver.Firefox()
        driver = webdriver.Chrome() # IF YOU ARE USING CHROME.	
        driver.maximize_window()

        for person_name in self.data:
            print('线程：', self.name, ':', person_name, '\n')
#            if person_name in ['funeral', 'cancer']:
#                continue
            # https://www.gettyimages.com/photos/john-travolta?family=editorial&phrase=john%20travolta&sort=mostpopular#license
            current_index = 0
            our_dir = os.path.join(r'.\getty',person_name) 
            if not os.path.isdir(our_dir): # If the folder does not exist in working directory, create a new one.
                os.makedirs(our_dir)
            else:
                all_index = len(os.listdir(our_dir))
                print(person_name, ' all index ', all_index / 2)
                if all_index / 2 >= 4990:
                    continue
                current_index = int(all_index // 2)
            url = 'https://www.gettyimages.com/photos/{0}?family=editorial&page={1}&phrase={2}&sort=newest#license'.format(person_name, int(current_index // 60) + 1, person_name)
            driver.get(url)
            browse_page(driver, person_name, 5000, our_dir, current_index, self.name)
        print ("退出线程：" + self.name)


    
if __name__ == '__main__':
    
    data = pd.read_csv(r'.\keys.txt', encoding='utf-8', sep='\t', header=None)
    data = np.array(data.iloc[:,0].tolist())

    print_info = ['']*9
    a2 = [ 'paradise', 'loved' , 'humor', 'laughter']
    thread_list = list()
    for i in range(0, 4):
        thread1 = myThread(i, [a2[i]])
        thread1.start()
        thread_list.append(thread1)
#    for i in list(range(96,101)):
#        thread1 = myThread(i, data[i:i+1])
#        thread1.start()
#        thread_list.append(thread1)
        

    for thread in thread_list:
        thread.join()
    

#    driver = webdriver.Chrome() # IF YOU ARE USING CHROME.	
#    driver.minimize_window()
#
#    for person_name in data:
##        if person_name in ['funeral', 'cancer']:
##            continue
#        # https://www.gettyimages.com/photos/john-travolta?family=editorial&phrase=john%20travolta&sort=mostpopular#license
#        url = 'https://www.gettyimages.com/photos/funeral?family=editorial&phrase={}&sort=newest#license'.format(person_name)
#        our_dir = os.path.join(r'D:\ddd_article2\getty1',person_name)
#        driver.get(url)
#        if not os.path.isdir(our_dir): # If the folder does not exist in working directory, create a new one.
#            os.makedirs(our_dir)
#        browse_page(driver, person_name, 5000, our_dir, 1)		
