import socket
import TextSummariser
import time
#Solution from https://pymotw.com/2/socket/tcp.html

#This method works by first getting the word embeddings from the GloVe text file as this process takes a while so it's better to get this step done first. It then listens on port 9878 for incoming connections.
#After connecting with a client( C# book data collector console app), it then reads all the data being sent from the client which is the description of the book.
#If continues to read from the network stream until is recieves a "#" from the client which tells the server that all the data has been received from the client.It
#then decodes the receives data into a string and uses the TextSummariser class to summarise the book description. After it receives the summarised book descrption, it then
#encodes it and sends it back to the client. After it sends it, it also sends "#" symbol to tell the client that the server has sent all the data. The server then wait until
#the client initiates the process for closing connection via a partial handshake. After closing connection, the whole process repeats.
def listen(word_embeddings):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('10.40.55.194', 9878)
    sock.bind(server_address)
    sock.listen(1)

    while True:
        print('Waiting for connection')
        connection, client_address = sock.accept()

        try:
            while True:
                bookDescriptionData = connection.recv(1024)

                #If the "#" symbol is received in the data then that implies that all the data has been received from the client
                if bookDescriptionData.decode("utf-8"):
                    summarisedBookDescription = TextSummariser.summarise(bookDescriptionData.decode("utf-8"),word_embeddings)

                    if summarisedBookDescription:
                        connection.sendall((summarisedBookDescription).encode("utf-8"))
                        time.sleep(5)
                        connection.sendall(("#").encode("utf-8"))
                        print('Sent summarised book description')
                    break

            while True:
                data = connection.recv(1024)

                if "FIN" in data.decode("utf-8"):
                    print('Received ACKFIN')
                    connection.sendall(("ACKFIN").encode("utf-8"))
                    print('Sent ACKFIN')

                elif "ACK" in data.decode("utf-8"):
                    print('Received ACK')
                    connection.sendall(("CLOSED").encode("utf-8"))
                    break

        finally:
            time.sleep(1)
            print('Closing connection')
            connection.close()
