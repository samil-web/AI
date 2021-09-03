# Counting wood.
text = """How much wood would a woodchuck chuck
If a woodchuck could chuck wood?
He would chuck, he would, as much as he could,
And chuck as much as a woodchuck would
If a Mr. Smith could chuck wood\n\r\t."""
def wood_counter(text):
    text = text.replace("?", "").replace(".","")
    l = text.lower().strip().split()
    counter = 0
    for word in l:
        if word == "wood":
            counter +=1
    return counter
    
wood_counter(text)


