# top level module for PicMe 
from scraper import *
from imageUtil import *
from imageprocess import *

if __name__ == "__main__":
    scraper = Scraper()
    codesFromFile = scraper.parseFlags(len(sys.argv), sys.argv)
    if (codesFromFile):
        scraper.extractShortCodesFromCsv(sys.argv[2])
    else:
        scraper.extractShortCodesFromJson(sys.argv[2])
    posts = [] # list of post objects

    # getting all posts given short code
    for shortcode in scraper.shortcodes:
        newPost = scraper.getPostRequestBody(shortcode)
        if newPost != None:
            posts.append(scraper.getPostRequestBody(shortcode))

    # TODO: unify the loop below with the loop above
    # are there any disadvantages? Raul's opinion: I like how cleaner and 
    # decomposed the code looks, but that might incur in inefficient code

    postImg = Image(posts[0].getImgUrl(), True)
    extractSectorsFeature(postImg, 30, 30, True)


        
