
import os
from colorama import Fore
from pathlib import Path

list_of_created = []


def colorIs(i):
    if (i % 5 == 0):
        return Fore.RED
    elif (i % 5 == 1):
        return Fore.GREEN
    elif (i % 5 == 2):
        return Fore.YELLOW
    elif (i % 5 == 3):
        return Fore.CYAN
    else:
        return Fore.MAGENTA


def MainCranField(path):
    Documents = os.listdir(path)
    document = Documents[0]
    blockSize = 100
    current = 1
    list_of_created.append("doc1.txt")
    title = ""
    author = ""
    publications = ""
    text = ""
    prev = ""
    docId = ""
    document_created = 1
    with open(path + "/" + document, 'r', encoding='utf-8') as file:
        prev = ""
        for line in file:
            line = line.lstrip(" ")
            line = line.removesuffix("\n")
            if ".I" in line:
                if current != 1:
                    with open(f"doc{document_created}.txt", "a") as f:
                        f.write(docId + " --- " + title + " --- " + author + " --- " + publications + " --- " + text)
                        f.write("\n")
                title = ""
                author = ""
                publications = ""
                text = ""
                prev = ""
                line = line.lstrip(" ")
                listIs = line.split(" ")
                currentLine = listIs[1].removesuffix("\n")
                docId = currentLine
                current += 1
                if current >= blockSize:
                    list_of_created.append(f"doc{document_created + 1}.txt")
                    document_created += 1
                    current = 1
            elif ".T" in line:
                prev = "T"
                continue
            elif ".A" in line:
                prev = "A"
                continue
            elif ".B" in line:
                prev = "B"
                continue
            elif ".W" in line:
                prev = "W"
                continue

            if prev == 'T':
                if len(title) != 0: title += " "
                title += line
            elif prev == 'A':
                if len(author) != 0: author += " "
                author += line
            elif prev == 'B':
                if len(publications) != 0: publications += " "
                publications += line
            elif prev == 'W':
                if len(text) != 0: text += " "
                text += line
    with open(f"doc{document_created}.txt", "a") as f:
        f.write(docId + " --- " + title + " --- " + author + " --- " + publications + " --- " + text)
        f.write("\n")


print("==========================================================================")
print(f"{Fore.MAGENTA} \t\t\t\tWelcome to Boolean retrieval using BSBI! \t\t")
print("---------------------------------------------------------------------------")
print(f"{Fore.GREEN}  Please enter the folder location where document is present: ")
path = input()
value = os.path.isfile("merged.txt")
if value == False :
    MainCranField(path)
    with open("merged.txt", "a") as new_created_file:
        for name in list_of_created:
            with open(name, "r") as file:
                for line in file:
                    new_created_file.write(line)
    for name in list_of_created:
        os.remove(name)
else:
    print("ALready computed!!")
print()
print(f"{Fore.CYAN}__________________________________________________________________________")
print(f"{Fore.CYAN}| 1.   Search the documents based on the publication year and publisher  |")
print(f"{Fore.CYAN}| 2.   Search the documents based on the author and the published year   |")
print(f"{Fore.CYAN}| 3.   Rank the documents based on the given weights and input           |")
print(f"{Fore.CYAN}__________________________________________________________________________")
print(f"{Fore.RED}Choose the sent from above table to perform the operation or -1 to exit : ")
sent = int(input())
while sent != -1:
    if sent == 1:
        published_year = input(f"{Fore.YELLOW}Enter the publication year : ")
        publisher_name = input(f"{Fore.YELLOW}Enter the name of the publisher : ")
        listOf = []
        with open("merged.txt", "r") as file:
            for line in file:
                listIs = line.split(" --- ")
                listIs[4].removesuffix("\n")
                if publisher_name in listIs[3] and published_year in listIs[3]:
                    listFor = [listIs[0], listIs[2], listIs[1], published_year]
                    listOf.append(listFor)
        listOf = sorted(listOf, key=lambda x: x[3])
        print("===========================================================================================")
        print("\tDocId\t-\tAuthors\t-\tPublished Year\t-\tTitle")

        for i in range(len(listOf)):
            print(f"{colorIs(i)}{listOf[i][0]} - {listOf[i][1]} - {listOf[i][3]} - {listOf[i][2]}")
    elif sent == 2:
        published_year = input(f"{Fore.YELLOW}Enter the publication year : ")
        author_name = input(f"{Fore.YELLOW}Enter the name of the publisher : ")
        listOf = []
        with open("merged.txt", "r") as file:
            for line in file:
                listIs = line.split(" --- ")
                listIs[4].removesuffix("\n")
                if author_name in listIs[2] and published_year in listIs[3]:
                    listFor = [listIs[0], listIs[2], listIs[1], published_year]
                    listOf.append(listFor)
        listOf = sorted(listOf, key=lambda x: x[3])
        print("===========================================================================================")
        print("\tDocId\t-\tAuthors\t-\tPublished Year\t-\tTitle")

        for i in range(len(listOf)):
            print(f"{colorIs(i)}{listOf[i][0]} - {listOf[i][1]} - {listOf[i][3]} - {listOf[i][2]}")
    elif sent == 3:
        print(f"{Fore.WHITE} How many documents should be ranked ?")
        rank = int(input())
        fetched = 0
        print("Give the weights in such a way that they should add up to 1 and in range [0,1]")
        titleWeight = float(input("Enter the weight to be given to the title : "))
        textWeight = float(input("Enter the weight to be given to the text   : "))
        nameIs = input("Enter the words to be searched : ")
        fName = nameIs.split(" ")
        with open("merged.txt","r") as file:
            listOf = []
            for line in file:
                score = 0
                listIs = line.split(" --- ")
                for name in fName:
                    if name in listIs[1]:
                        score += titleWeight
                    if name in listIs[4]:
                        score += textWeight
                if score > 0:
                    listFor = [listIs[0], listIs[2], listIs[1], score]
                    listOf.append(listFor)
                    fetched += 1
        listOf = sorted(listOf, key=lambda x: x[3],reverse=True)
        print("===========================================================================================")
        print("\tDocId\t-\tAuthors\t-\tTitle\t-\tScore")

        for i in range(min(len(listOf),rank)):
            print(f"{colorIs(i)}{listOf[i][0]} - {listOf[i][1]} - {listOf[i][2]} - {listOf[i][3]}")
        print()
        print()
        print(f"\t\t\t\t{fetched}/{1400} documents scored and {rank}/{fetched} are displayed")
    print(f"{Fore.RED}Choose the sent from above table to perform the operation or -1 to exit : ")
    sent = int(input())


print("===========================================================================================")
print(f"{Fore.MAGENTA} \t\t\t\t Conjuction, Disjuction and Negatation Operations \t\t")
print(f"{Fore.CYAN}_____________________")
print(f"{Fore.CYAN}| 1.   Disjuction   |")
print(f"{Fore.CYAN}| 2.   Conjuction   |")
print(f"{Fore.CYAN}| 3.   Negatation   |")
print(f"{Fore.CYAN}_____________________")
print(f"{Fore.RED}Choose the sent from above table to perform the operation or -1 to exit : ")
sent1 = int(input())
while sent1 != -1:
    if sent1 == 1:
        inputVal = input("Enter the simple disjuction : ")
        listIs_ = inputVal.split(" ")
        term1 = listIs_[0]
        term2 = listIs_[2]
        listOf = []
        weight_title = 0.75
        weight_text = 0.25
        fetched = 0
        with open("merged.txt","r") as file:
            listOf = []
            for line in file:
                score = 0
                listIs = line.split(" --- ")
                if term1 in listIs[1]:
                    score += weight_title
                if term2 in listIs[1]:
                    score += weight_title
                if term1 in listIs[4]:
                    score += weight_text
                if term2 in listIs[4]:
                    score += weight_text
                if score > 0:
                    listFor = [listIs[0], listIs[2], listIs[1], score]
                    listOf.append(listFor)
                    fetched += 1
            listOf = sorted(listOf, key=lambda x: x[3], reverse=True)
            print("===========================================================================================")
            print("\tDocId\t-\tAuthors\t-\tTitle\t-\tScore")

            for i in range(len(listOf)):
                print(f"{colorIs(i)}{listOf[i][0]} - {listOf[i][1]} - {listOf[i][2]} - {listOf[i][3]}")
            print()
            print()
            print(f"\t\t\t\t{fetched}/{1400} documents scored")

    elif sent1 == 2:
        inputVal = input("Enter the simple conjuction : ")
        listIs_ = inputVal.split(" ")
        term1 = listIs_[0]
        term2 = listIs_[2]
        listOf = []
        weight_title = 0.75
        weight_text = 0.25
        fetched = 0
        with open("merged.txt","r") as file:
            listOf = []
            for line in file:
                score = 0
                listIs = line.split(" --- ")
                if term1 in listIs[1] and term2 in listIs[1]:
                    score += weight_title
                if term1 in listIs[4] and term2 in listIs[4]:
                    score += weight_text
                if score > 0:
                    listFor = [listIs[0], listIs[2], listIs[1], score]
                    listOf.append(listFor)
                    fetched += 1
            listOf = sorted(listOf, key=lambda x: x[3], reverse=True)
            print("===========================================================================================")
            print("\tDocId\t-\tAuthors\t-\tTitle\t-\tScore")

            for i in range(len(listOf)):
                print(f"{colorIs(i)}{listOf[i][0]} - {listOf[i][1]} - {listOf[i][2]} - {listOf[i][3]}")
            print()
            print()
            print(f"\t\t\t\t{fetched}/{1400} documents scored")

    elif sent1 == 3:
        inputVal = input("Enter the simple negatation : ")
        listIs = inputVal.split(" ")
        term1 = listIs[0]
        term2 = listIs[2]
        listOf = []
        weight_title = 0.75
        weight_text = 0.25
        fetched = 0
        with open("merged.txt", "r") as file:
            listOf = []
            for line in file:
                score = 0
                listIs = line.split(" --- ")
                if term1 in listIs[1] and term2 not in listIs[1]:
                    score += weight_title
                if term1 in listIs[4] and term2 not in listIs[4]:
                    score += weight_text
                if score > 0:
                    listFor = [listIs[0], listIs[2], listIs[1], score]
                    listOf.append(listFor)
                    fetched += 1
            listOf = sorted(listOf, key=lambda x: x[3], reverse=True)
            print("===========================================================================================")
            print("\tDocId\t-\tAuthors\t-\tTitle\t-\tScore")

            for i in range(len(listOf)):
                print(f"{colorIs(i)}{listOf[i][0]} - {listOf[i][1]} - {listOf[i][2]} - {listOf[i][3]}")
            print()
            print()
            print(f"\t\t\t\t{fetched}/{1400} documents scored")
    else:
        break
    print()
    print()
    print(f"{Fore.CYAN}_____________________")
    print(f"{Fore.CYAN}| 1.   Disjuction   |")
    print(f"{Fore.CYAN}| 2.   Conjuction   |")
    print(f"{Fore.CYAN}| 3.   Negatation   |")
    print(f"{Fore.CYAN}_____________________")
    print(f"{Fore.RED}Choose the sent from above table to perform the operation or -1 to exit : ")
    sent1 = int(input())