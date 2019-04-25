import requests as req
from bs4 import BeautifulSoup


base_url = "https://www.swiggy.com/"
loc = input("enter your location \t:\t")
append_url = "/restaurants?filters=%7B%22SHOW_RESTAURANTS_WITH%22%3A%5B%22Pure+Veg%22%5D%7D&sortBy=RELEVANCE"
print()


def soupify(loc="thrissur"):
    """

    :type loc: string
    """
    url = base_url + str(loc) + append_url
    print(url)
    swigger = req.get(url)
    # print(swigger.status_code)  # ,swigger.raw,swigger.url)

    soup = BeautifulSoup(swigger.text, 'html.parser')
    print(soup)
    bois = soup.findAll('div',{'class':'MZy1T'})
    print(bois)
    # file_path = f'C:/Users/ggmah/Desktop/swiggy_{loc}.txt'
    # # print("*")
    # # with open(file_path,"w+") as fp:
    #     # print('**')
    #     # for boi in bois:
    #     #     print(boi)

if __name__ == "__main__":
    soupify(loc)
