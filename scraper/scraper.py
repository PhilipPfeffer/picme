from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
import requests
import re
import random

ALERT_CLASS_NAME="mt3GC" ## TODO - config file to store all these constants
POST_OBJECT_CLASS_NAME="_8Rm4L M9sTE  L_LMM SgTZ1   ePUX4"
USERNAME_CLASS_NAME="FPmhX notranslate nJAzx"
class Scraper:
    def __init__(self): 
        self.driver = webdriver.Chrome()
        self.supportedImgFileTypes = ["image/jpeg"]
        ## hardcoding cookies for now
        self.cookies = [{"name": "sessionid", "value": "23063648597%3AIsOfSAvrrcF6ww%3A4"}]
        return

    def connectToWebsite(self, url): #, cookies): TODO NOT WORKING CONSISTENTLY
        self.url = url
        try:
            self.driver.get(url)
            [self.driver.add_cookie(cookie) for cookie in self.cookies]
            self.driver.refresh
            return True
        except Exception  as e:
            print(e)
            return False 

    def dismissInstagramNotificationAlert(self): # WORKS!!
        try:
            alert = self.driver.find_element_by_class_name("mt3GC")
            alert.click()
            return True
        except Exception as e:
            print(e)
            return False

    def getAllPostsInScreen(self):
        try:
            xpath = "//article[@class=\'"+POST_OBJECT_CLASS_NAME+"\']"
            posts = self.driver.find_elements_by_xpath(xpath)
            return posts
        except Exception as e:
            print(e)
            return None

    def getImagesSaveLocally(self, post): ##  UNSTABLE
        try:
            # get img data
            imgHtml = post.find_element_by_class_name("FFVAD")
            rawUrl  = imgHtml.get_attribute("srcset")
            parsedUrl = re.search(r'https[^\s]*', rawUrl)
            imgUrl = parsedUrl.group()
            r = requests.get(imgUrl)
            if (r.headers["Content-Type"] not in self.supportedImgFileTypes): return
            name = "img"+str(random.randint(0,1000))+".jpg"
            open(name, "wb").write(r.content)
            return r.content
        except Exception as e:
            print(e)
            return []

    def getUsernameFromPost(self, post):
        try:
            # get username of account that posted
            xpath = "//a[@class=\'"+USERNAME_CLASS_NAME+"\']"
            usernameHtml = post.find_element_by_xpath(xpath)
            return usernameHtml
            #open("testimg.png", "wb").write(r.content)
        except Exception as e:
            print(e)
            return []

class Post():
    def __init__(self):
        return

    def getImage(self):
        raise NotImplementedError 

    def setImage(self, image):
        raise NotImplementedError 

    def getUsername(self):
        raise NotImplementedError 

    def setUsername(self, un)
        raise NotImplementedError

    def getLikeCount(self):
        raise NotImplementedError 

    def setLikeCount(self, lc):
        raise NotImplementedError


if __name__  == "__main__":
    instaUrl = "https://www.instagram.com"
    scraper = Scraper()
    
