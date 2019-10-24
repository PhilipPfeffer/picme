from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
import requests

ALERT_CLASS_NAME="mt3GC" ## TODO - config file to store all these constants
POST_OBJECT_CLASS_NAME="_8Rm4L M9sTE  L_LMM SgTZ1   ePUX4"
class Scraper:
    def __init__():
        self.driver = webdriver.Chrome()
        self.supportedImgFileTypes = ["image/jpeg"]
        return

    def connectToWebsite(self, url, cookies):
        self.url = url
        try:
            self.driver.get(url)
            self.driver.add_cookie(cookie)
            self.driver.refresh
            return True
        except:
            print("An exception was thrown while connecting to website")
            return False

    def dismissInstagramNotificationAlert(self):
        try:
            alert = self.driver.find_element_by_class_name("mt3GC")
            alert.click()
            return True
        except:
            print("An exception was thrown while dismissing alert")
            return False

    def getAllPostsInScreen(self):
        try:
            posts = driver.find_elements_by_class_name(POST_OBJECT_CLASS_NAME)
            return posts
        except:
            print("Could not fetch posts")
            return None

    def getImagesSaveLocally(self): 
        try:
            imgs = driver.find_elements_by_class_name("FFVAD")
            for img in imgs:
                img.get_attribute("srcset")
                matched = re.search(r'https[^\s]*',toparse)
                linkToImg = matched.group()
                r = requests.get(linkToImg)
                if (r.headers["Content-Type"] not in self.supportedImgFileTypes): return
                open("testimg.png", "wb").write(r.content)
                ## continue on and get further data on the post
                # username, date, num of likes, num of comments, etc...
