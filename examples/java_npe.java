// The legendary NullPointerException
import java.util.HashMap;
import java.util.Map;

public class java_npe {
    private Map<String, User> userCache = new HashMap<>();

    static class User {
        String name;
        Address address;

        User(String name) {
            this.name = name;
            // Forgot to initialize address!
        }
    }

    static class Address {
        String city;
        String country;
    }

    public User getUser(String id) {
        // Returns null if not found
        return userCache.get(id);
    }

    public String getUserCity(String userId) {
        User user = getUser(userId);
        // NPE on user.address (user is null) or user.address.city (address is null)
        return user.address.city;
    }

    public static void main(String[] args) {
        java_npe app = new java_npe();
        // User "123" doesn't exist in cache
        String city = app.getUserCity("123");
        System.out.println("User lives in: " + city);
    }
}
