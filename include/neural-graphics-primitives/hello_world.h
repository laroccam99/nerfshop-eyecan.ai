#include <iostream>
#include <string>

class hello_world {
private:
    std::string privateString; // Variabile stringa privata

public:
    // Costruttore che inizializza la variabile privata
    hello_world() {
        privateString = "Hello world";
    }

    // Metodo pubblico per ottenere la variabile privata (getter)
    std::string getPrivateString() const {
        return privateString;
    }
};    