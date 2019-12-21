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
            print(len(self.shortcodes))
        except Exception as e:
            if (resp.status_code == 429):
                print("Waiting for a minute because one 429 from Instagram won't stop us.")
                time.sleep(60)
                print("Retrying request")

                try:
                    resp = requests.get(url)
                    resp.raise_for_status()
                    jsonData = json.loads(resp.text)
                    posts = jsonData["graphql"]["hashtag"]["edge_hashtag_to_top_posts"]["edges"]
                    for post in posts:
                        self.shortcodes.append(post["node"]["shortcode"])
                    print(len(self.shortcodes))
                except Exception as e:
                    print("extractShortCodesFromJson: Retrial failed: " + str(e))
            else:
                print("extractShortCodesFromJson: Error: " + str(e))

    def extractShortCodesFromCsv(self, filename):
        with open(filename) as f:
            self.shortcodes = [row["shortcode"] for row in csv.DictReader(f)]

    def extractHashtagsFromCsv(self, filename):
        with open(filename) as f:
            self.hashtags = [row["hashtags"] for row in csv.DictReader(f)]
        return self.hashtags

    def extractUsernamesFromCsv(self, filename):
        with open(filename) as f:
            self.usernames = [row["username"] for row in csv.DictReader(f)]
        return self.usernames

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

    def extractMostRecentPosts(self, user):
        try:
            print(f"username: {user}")
            url = "https://www.instagram.com/"+str(user)+"/?__a=1"
            resp = requests.get(url)
            resp.raise_for_status()
            jsonData = json.loads(resp.text)
            if jsonData["graphql"]["user"]["is_joined_recently"]:
                raise  ValueError("too soon")
            hasChannel  = jsonData["graphql"]["user"]["has_channel"]
            isVerified = jsonData["graphql"]["user"]["is_verified"]
            followers = jsonData["graphql"]["user"]["edge_followed_by"]["count"]
            follows = jsonData["graphql"]["user"]["edge_follow"]["count"]
            isBusinessAcc = jsonData["graphql"]["user"]["is_business_account"]
            posts = jsonData["graphql"]["user"]["edge_owner_to_timeline_media"]["edges"]
            totalLikes = 0
            refinedPosts = []
            for post in posts:
                postdata = post["node"]
                isVideo = postdata["is_video"]
                if isVideo: 
                    continue
                timestamp = postdata["taken_at_timestamp"]
                inLast24H = (time.time() - timestamp) < 86400
                inLastYr = (time.time() - timestamp) < 31536000
                if (inLast24H or not inLastYr): 
                    continue
                shortcode = postdata["shortcode"]
                caption_container  = postdata["edge_media_to_caption"]["edges"]
                if len(caption_container) > 0:
                    caption = caption_container[0]["node"]["text"]
                else:
                    caption  = ""
                numComments = postdata["edge_media_to_comment"]["count"]
                numLikes = postdata["edge_liked_by"]["count"]
                totalLikes += numLikes 
                imgUrl = postdata["thumbnail_resources"][2]["src"]
                accessibilityCaption = postdata["accessibility_caption"]
                refinedPosts.append([caption, user, shortcode, timestamp, imgUrl, numLikes, numLikes/followers, numComments, numComments/followers, accessibilityCaption])
            if (len(refinedPosts) == 0):
                return
            userAverageLikes = totalLikes/len(refinedPosts)
            postObjects = []
            for rp in refinedPosts:
                caption, user, shortcode, timestamp, imgUrl, numLikes, likeRatio, numComments, commentRatio, accessibilityCaption = rp
                p = Post(caption, user, userAverageLikes, shortcode, timestamp, imgUrl, numLikes, likeRatio, numComments, commentRatio, accessibilityCaption, hasChannel,isVerified,followers,follows,isBusinessAcc)
                postObjects.append(p)
            return postObjects

        except Exception as e:
            print("extractMostRecentPosts: User error: " + str(e))
            return None


class Post:
    def __init__(self, caption, username, userAverageLikes, shortcode, timestamp, imgUrl, likeCount, likeRatio, commentCount, commentRatio, accessibilityCaption, hasChannel,isVerified,followers,follows,isBusinessAcc):
        self.__caption = caption
        self.__username = username
        self.__userAverageLikes = userAverageLikes
        self.__shortcode = shortcode
        self.__timestamp = timestamp
        self.__imgUrl = imgUrl
        self.__likeCount = likeCount
        self.__likeRatio = likeRatio
        self.__commentCount = commentCount
        self.__commentRatio = commentRatio
        self.__accessibilityCaption = accessibilityCaption
        self.__hasChannel = hasChannel  
        self.__isVerified = isVerified  
        self.__followers = followers  
        self.__follows = follows  
        self.__isBusinessAcc = isBusinessAcc  

    def __str__(self): #TODO UPDATE
        return (f"""Image object:\n\tcaption: {self.__caption},\n\t
username: {self.__username},\n\t
userAverageLikes: {self.__userAverageLikes},\n\t
shortcode: {self.__shortcode},\n\t
timestamp: {self.__timestamp},\n\t
imgUrl: {self.__imgUrl},\n\t
likeCount: {self.__likeCount},\n\t
likeRatio: {self.__likeRatio},\n\t
commentCount: {self.__commentCount},\n\t
commentRatio: {self.__commentRatio},\n\t
accessibilityCaption: {self.__accessibilityCaption},\n\t""")
    
    def getHeaderList():
        return ["caption", "username", "userAverageLikes", "shortcode", "timestamp", "imgUrl", "likeCount", "likeRatio", "commentCount", "commentRatio", "accessibilityCaption", "hasChannel","isVerified","followers","follows","isBusinessAcc"]

    def toList(self):
        return [self.__caption,self.__username,self.__userAverageLikes,self.__shortcode,self.__timestamp,self.__imgUrl,self.__likeCount,self.__likeRatio,self.__commentCount,self.__commentRatio,self.__accessibilityCaption, self.__hasChannel,self.__isVerified,self.__followers,self.__follows,self.__isBusinessAcc]

    def getCaption(self):
        return self.__caption
    def getUsername(self):
        return self.__username
    def getUserAverageLikes(self):
        return self.__userAverageLikes
    def getShortcode(self):
        return self.__shortcode
    def getTimestamp(self):
        return self.__timestamp
    def getImgUrl(self):
        return self.__imgUrl
    def getLikeCount(self):
        return self.__likeCount
    def getLikeRatio(self):
        return self.__likeRatio
    def getCommentCount(self):
        return self.__commentCount
    def getCommentRatio(self):
        return self.__commentRatio
    def getAccessibilityCaption(self):
        return self.__accessibilityCaption


if __name__  == "__main__":
    scraper = Scraper()
    if (len(sys.argv) < 2):
        print("Usage: python3 scraper.py  [-h | -u] filename")
    if (sys.argv[1] == "-h"): #scrape by hashtag
        hashtagsFromCsv = scraper.extractHashtagsFromCsv(sys.argv[2])
        for hashtag in hashtagsFromCsv:
            print(f"hashtag: {hashtag}")
            url = "https://www.instagram.com/explore/tags/"+str(hashtag)+"/?__a=1"
            scraper.extractShortCodesFromJson(url)
        posts = []
        for shortcode in scraper.shortcodes:
            newPost = scraper.getPostRequestBody(shortcode)
            if newPost != None:
                posts.append(scraper.getPostRequestBody(shortcode))
        scraper.writeToCsv(f"datasets/{sys.argv[2][:-4]}FULL.csv", posts)
    elif (sys.argv[1] == "-u"): # scrape by username
        usernames = scraper.extractUsernamesFromCsv(sys.argv[2])
        allPosts = []
        for user in usernames:
            recentPosts = scraper.extractMostRecentPosts(user)
            if recentPosts != None:
                for post in recentPosts:
                    allPosts.append(post)
            scraper.writeToCsv(f"datasets/{sys.argv[2][:-4]}FULL.csv", allPosts)
    else:
        print("Invalid flag.\nUsage: python3 scraper.py  [-h | -u]")