// TypeScript strict mode type error

interface User {
    id: number;
    name: string;
    email: string;
}

interface ApiResponse {
    data: User | null;
    error?: string;
}

function fetchUser(id: number): ApiResponse {
    // Simulating API that returns null for unknown users
    if (id !== 1) {
        return { data: null, error: "User not found" };
    }
    return {
        data: { id: 1, name: "Alice", email: "alice@example.com" }
    };
}

function sendWelcomeEmail(user: User): void {
    console.log(`Sending welcome email to ${user.email}`);
}

function main(): void {
    const response = fetchUser(999);

    // TypeScript error: Argument of type 'User | null' is not assignable
    // to parameter of type 'User'
    sendWelcomeEmail(response.data);
}

main();
