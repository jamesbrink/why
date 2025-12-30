#!/usr/bin/env ruby
# NoMethodError on nil - Ruby's classic gotcha

class UserRepository
  def initialize
    @cache = {}
  end

  def find(id)
    # Simulating a cache miss - returns nil
    @cache[id]
  end
end

class UserService
  def initialize
    @repo = UserRepository.new
  end

  def get_user_email(user_id)
    user = @repo.find(user_id)
    # Boom! user is nil, can't call .email on it
    user.email
  end
end

service = UserService.new
puts service.get_user_email(42)
