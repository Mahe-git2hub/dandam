def printHang(n):
    gal = [['---- '],
           ['|  | '],
           ['|    '],
           ['|    '],
           ['|    ']]
    if n < 6:
        gal[2] = ['|  o ']
    if n < 5:
        gal[3] = ['| /  ']
    if n < 4:
        gal[3] = ['| / \\']
    if n < 3:
        gal[3] = ['| /|\\']
    if n < 2:
        gal[4] = ['| /  ']
    if n < 1:
        gal[4] = ['| / \\']
    for i in gal:
        print(''.join(i))




def word_pick():
    import random
    return random.choice(open(r'C:\Users\ggmah\Desktop\words.txt').read().split()).strip()



if __name__=="__main__":
    
    answer='y'
    
    while answer!='n' and answer!='N':
        
        new_word = word_pick()
    
        hidden_word = ""
    
        #for i in new_word:
            #print (i + " ", end=" ")
    
        print ("HANGMAN!\n")
        print ("You have 6 incorrect tries to guess the letters. (USE CAPS LOCK!)\n")
        for i in range(len(new_word)):
            print ("_" + " ", end=" "),
        
        chance = True
    
        count = 0
    
        used=[]
    
        while (chance==True and count<6):
        
            guess = input("Enter your guess: ")
        
            if guess in used:
                print ("Already guessed that one!")
           
            side_word=""
              
            for i in new_word:
                if  guess==i:
                    side_word += (i + "  ")
                
                elif i in used:
                    side_word += (i + "  ")
            
                elif guess in used:
                    side_word += "_  "
                                                       
                else:
                    side_word += "_  "
       
            if guess in new_word:
                if used.count(guess)<1:
                    used.append(guess)
        
            if guess not in new_word:
                count+=1
                printHang((6-count))
        
                
            print("The word is...")
            print(side_word)
        
            hidden_word = side_word
        
            if "_" in hidden_word:
                chance = True
            else:
                chance = False
    
        hidden_word = [x for x in hidden_word if x.isalpha()]
        hd=""
        for k in hidden_word:
            hd += k
    
        if hd==new_word:
            print ("You won!")
            print ("Won with %s tries remaining."%(6-count))
        else:
            print ("Incomplete!")
            print ("The word was ",new_word)
        
        answer=input("Do you wanna play again (y/n)? ")
