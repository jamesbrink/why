/* Classic NULL pointer dereference - the OG of segfaults */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct User {
    int id;
    char *name;
    struct User *manager;
};

struct User* find_user(int id) {
    /* Simulating user not found */
    if (id != 1) {
        return NULL;
    }
    struct User *user = malloc(sizeof(struct User));
    user->id = 1;
    user->name = "Alice";
    user->manager = NULL;
    return user;
}

void print_manager_name(struct User *user) {
    /* Forgot to check if manager is NULL! */
    printf("Manager: %s\n", user->manager->name);
}

int main() {
    struct User *user = find_user(1);

    if (user) {
        printf("Found user: %s\n", user->name);
        print_manager_name(user);  /* Segfault here! */
    }

    return 0;
}
