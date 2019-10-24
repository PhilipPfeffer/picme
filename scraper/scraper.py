from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert

ALERT_CLASS_NAME="mt3GC" ## TODO - config file to store all these constants

class Scraper:
    def __init__():
        self.driver = webdriver.Chrome()
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

    

