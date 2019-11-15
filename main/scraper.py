#from selenium import webdriver
#from selenium.webdriver.common.keys import Keys
#from selenium.webdriver.common.alert import Alert
import requests
import re
import random
import sys
import csv
import json
import pdb
from html.parser import HTMLParser
from bs4 import BeautifulSoup
import time

ALERT_CLASS_NAME="mt3GC" ## TODO - config file to store all these constants
POST_OBJECT_CLASS_NAME="_8Rm4L M9sTE  L_LMM SgTZ1   ePUX4"
USERNAME_CLASS_NAME="FPmhX notranslate nJAzx"
INSTA_BASEURL = "https://www.instagram.com/"

class InstaUtil:
    def getNumFollowers(username):
        try:
            resp = requests.get(INSTA_BASEURL + username + "/?__a=1")
            resp.raise_for_status()
            jsonData = json.loads(resp.text)
            followerCount = jsonData["graphql"]["user"]["edge_followed_by"]["count"]
            return followerCount
        except Exception as e:
            raise ValueError("InstaUtil.getNumFollowers: Error: " + str(e))


class Scraper:
    def __init__(self):
        self.shortcodes =  []
        self.hashtags = []
        self.instaBaseUrl = "https://www.instagram.com/"

    def parseFlags(self, argc, argv): # argv is sys.argv, a list of command line arguments
    # returns mode
        if (argc != 3):
            print("scraper.py takes 2 arguments\nUsage: python3 scraper.py -f <shortcode-csv-file> | -u <instagram-url>")
            sys.exit(0)
        if (argv[1] not in ["-f", "-u"]):
            print("Incorrect flag\nUsage: python3 scraper.py -f <shortcode-csv-file> | -u <instagram-url>")    
            sys.exit(0)
        if (argv[1] == "-f"):
            if (argv[2][len(argv[2])-4:] != ".csv"):
                print("Usage: python3 scraper.py <shortcode-list-file>\nError: file has to be a csv table.")
                sys.exit(0)
            return True
        else: # flag is "-u"
            if ((argv[2][:39] != "https://www.instagram.com/explore/tags/") or (argv[2][-6:] != "?__a=1")):
                print("Invalid url. Correct url format should be https://www.instagram.com/explore/tags/\{hashtag\}/?__a=1")
                sys.exit(0)
            return False

    def extractShortCodesFromJson(self, url):
        # sample url - https://www.instagram.com/explore/tags/food/?__a=1
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            jsonData = json.loads(resp.text)
            posts = jsonData["graphql"]["hashtag"]["edge_hashtag_to_top_posts"]["edges"]
            for post in posts:
                self.shortcodes.append(post["node"]["shortcode"])
            print(self.shortcodes)
        except Exception as e:
            print("extractShortCodesFromJson: Error parsing: " + str(e))

    def extractShortCodesFromCsv(self, filename):
        with open(filename) as f:
            self.shortcodes = [row["shortcode"] for row in csv.DictReader(f)]

    def extractHashtagsFromCsv(self, filename):
        with open(filename) as f:
            self.hashtags = [row["hashtags"] for row in csv.DictReader(f)]
        return self.hashtags

    def getPostRequestBody(self, shortcode):
        try:
            resp = requests.get(self.instaBaseUrl + "p/" + shortcode)
            resp.raise_for_status()
            respText = resp.text
            data = re.search(r'<script type="text/javascript">window\._sharedData =[^<]*', respText).group()
            data = data[52:-1]
            jsonData = json.loads(data)
            shortcode_media = jsonData["entry_data"]["PostPage"][0]["graphql"]["shortcode_media"]
            caption = shortcode_media["edge_media_to_caption"]["edges"][0]["node"]["text"]
            username = shortcode_media["owner"]["username"]
            fullname = shortcode_media["owner"]["full_name"]
            shortcode = shortcode_media["shortcode"]
            timestamp = shortcode_media["taken_at_timestamp"]
            imgList = shortcode_media["display_resources"]
            imgUrl = imgList[len(imgList) - 1]["src"]
            likeCount = shortcode_media["edge_media_preview_like"]["count"]
            commentCount = shortcode_media["edge_media_preview_comment"]["count"]
            isVideo = shortcode_media["is_video"] == "true"
            isAd = shortcode_media["is_ad"] == "true"
            hashtags = []
            soup = BeautifulSoup(respText, 'html.parser')
            hashtags = []
            hashtagTags = soup.find_all("meta", property="instapp:hashtags")
            if hashtagTags != None:
                for h in hashtagTags:
                    hashtags.append(h["content"])
            numFollowers = InstaUtil.getNumFollowers(username)
            likeRatio = likeCount/numFollowers
            commentRatio = commentCount/numFollowers
            newPost = Post(caption, username, fullname, shortcode, timestamp, imgUrl, likeCount, likeRatio, commentCount, commentRatio, isVideo, isAd, hashtags)
            return newPost
        except Exception as e:
            print("getPostRequestBody: Error parsing: " + str(e))
            return None

    def writeToCsv(self, filename, posts):
        with open(filename, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(Post.getHeaderList())
            for post in posts:
                writer.writerow(post.toList())


class Post:
    def __init__(self, caption, username, fullname, shortcode, timestamp, imgUrl, likeCount, likeRatio, commentCount, commentRatio, isVideo, isAd, hashtags):
        self.__caption = caption
        self.__username = username
        self.__fullname = fullname
        self.__shortcode = shortcode
        self.__timestamp = timestamp
        self.__imgUrl = imgUrl
        self.__likeCount = likeCount
        self.__likeRatio = likeRatio
        self.__commentCount = commentCount
        self.__commentRatio = commentRatio
        self.__isVideo = isVideo
        self.__isAd = isAd
        self.__hashtags = hashtags

    def __str__(self):
        return (f"""Image object:\n\tcaption: {self.__caption},\n\tusername: {self.__username},\n\tfullname: {self.__fullname},\n\tshortcode: {self.__shortcode},\n\ttimestamp: {self.__timestamp},\n\timgUrl: {self.__imgUrl},\n\tlikeCount: {self.__likeCount},\n\tcommentCount: {self.__commentCount},\n\tisVideo: {self.__isVideo},\n\tisAd: {self.__isAd}\n\thashtags: {self.__hashtags}""")
    
    def getHeaderList():
        return ["caption", "username", "fullname", "shortcode", "timestamp", "imgUrl", "likeCount", "likeRatio", "commentCount", "commentRatio", "isVideo", "isAd", "hashtags"]

    def toList(self):
        return [self.__caption, self.__username, self.__fullname, self.__shortcode, self.__timestamp, self.__imgUrl, self.__likeCount, self.__likeRatio, self.__commentCount, self.__commentRatio, self.__isVideo, self.__isAd, self.__hashtags]


    # getters (public)
    def getCaption(self):
        return self.__caption

    def getUsername(self):
        return self.__username

    def getFullname(self):
        return self.__fullname

    def getShortcode(self):
        return self.__shortcode

    def getTimestamp(self):
        return self.__timestamp

    def getImgUrl(self):
        return self.__imgUrl

    def getLikeCount(self):
        return self.__likeCount

    def getCommentCount(self):
        return self.__commentCount

    def getIsVideo(self):
        return self.__isVideo

    def getIsAd(self):
        return self.__isAd

    def getHashtags(self):
        return self.__hashtags

if __name__  == "__main__":
    scraper = Scraper()
    hashtagsFromCsv = scraper.extractHashtagsFromCsv(sys.argv[1])
    for hashtag in hashtagsFromCsv:
        print(f"hashtag: {hashtag}")
        url = "https://www.instagram.com/explore/tags/"+str(hashtag)+"/?__a=1"
        scraper.extractShortCodesFromJson(url)
    posts = []
    for shortcode in scraper.shortcodes:
        newPost = scraper.getPostRequestBody(shortcode)
        if newPost != None:
            posts.append(scraper.getPostRequestBody(shortcode))
    scraper.writeToCsv("datasets/theNEWgreatdataset.csv", posts)
    # scraper = Scraper()
    # codesFromFile = scraper.parseFlags(len(sys.argv), sys.argv)
    # if (codesFromFile):
    #     scraper.extractShortCodesFromCsv(sys.argv[2])
    # else:
    #     scraper.extractShortCodesFromJson(sys.argv[2])
    # posts = []
    # for shortcode in scraper.shortcodes:
    #     newPost = scraper.getPostRequestBody(shortcode)
    #     if newPost != None:
    #         posts.append(scraper.getPostRequestBody(shortcode))
    # scraper.writeToCsv("datasets/dataset" + str(int(time.time()))+ ".csv", posts)

    
    #old main function with deprecated selenium driver
        #instaUrl = "https://www.instagram.com"
        #print("Full selenium mode has not been implemented yet")
        #return 0
        #scraper.connectToWebsite(instaUrl)
        #scraper.connectToWebsite(instaUrl)
        #scraper.dismissInstagramNotificationAlert()
        #postsInScreen = scraper.getAllPostsInScreen();
        #formattedPosts = []
        #for post in postsInScreen:
        #    dictPost = {}
        #    dictPost["image"], dictPost["imageUrl"] = scraper.getImageBytes(post)
        #    dictPost["username"] = scraper.getUsernameFromPost(post).text
        #    formattedPosts.append(dictPost)
        #for p in formattedPosts:
        #    print("post: ", dictPost["imageUrl"], dictPost["username"])
        #scraper.closeScraper()

###########################################################
class SeleniumScraper: # DEPRECATED
    def __init__(self): 
        self.driver = webdriver.Chrome()
        self.supportedImgFileTypes = ["image/jpeg"]
        ## hardcoding cookies for now
        self.cookies = [{"name": "sessionid", "value": "23063648597%3AIsOfSAvrrcF6ww%3A4"}]

    def getDriver(self):
        return self.driver

    def connectToWebsite(self, url): #, cookies): TODO NOT WORKING CONSISTENTLY- issue: I have to run this twice: the first run gets me to the login page and the second one actually puts the cookies.
        self.url = url
        try:
            self.driver.get(url)
            for cookie in self.cookies:
                self.driver.add_cookie(cookie)
            sleep(1)
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

    def getAllPostsInScreen(self): # Full selenium mode
        try:
            xpath = "//article[@class=\'"+POST_OBJECT_CLASS_NAME+"\']"
            posts = self.driver.find_elements_by_xpath(xpath)
            return posts
        except Exception as e:
            print(e)
            return None

    def getImageBytes(self, post): ##  UNSTABLE
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
            return (r.content, imgUrl)
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


    def closeScraper(self):
        self.driver.close()

###########################################################