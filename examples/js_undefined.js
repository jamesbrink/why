#!/usr/bin/env node
// Classic undefined property access chain

const user = {
    name: "Alice",
    profile: {
        email: "alice@example.com"
        // Note: settings is undefined
    }
};

function getUserTheme(user) {
    // This will fail because settings is undefined
    return user.profile.settings.theme;
}

function displayDashboard(user) {
    const theme = getUserTheme(user);
    console.log(`Loading dashboard with ${theme} theme...`);
}

displayDashboard(user);
