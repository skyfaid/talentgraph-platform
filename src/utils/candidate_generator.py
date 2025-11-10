"""
Generate realistic candidate names and emails for resumes.
"""
import random
from typing import Dict, Optional


# Common first names
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Kenneth", "Carol", "Kevin", "Amanda", "Brian", "Dorothy", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Jason", "Rebecca", "Edward", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Anna", "Stephen", "Brenda",
    "Larry", "Pamela", "Justin", "Emma", "Scott", "Nicole", "Brandon", "Helen",
    "Benjamin", "Samantha", "Samuel", "Katherine", "Gregory", "Christine", "Alexander", "Debra",
    "Patrick", "Rachel", "Frank", "Carolyn", "Raymond", "Janet", "Jack", "Catherine",
    "Dennis", "Maria", "Jerry", "Heather", "Tyler", "Diane", "Aaron", "Julie",
    "Jose", "Joyce", "Adam", "Victoria", "Nathan", "Kelly", "Henry", "Christina",
    "Douglas", "Joan", "Zachary", "Evelyn", "Kyle", "Judith", "Noah", "Megan",
    "Ethan", "Cheryl", "Jeremy", "Andrea", "Walter", "Hannah", "Christian", "Jacqueline",
    "Keith", "Martha", "Roger", "Gloria", "Terry", "Teresa", "Gerald", "Sara",
    "Harold", "Janice", "Sean", "Marie", "Austin", "Julia", "Carl", "Grace",
    "Arthur", "Judy", "Lawrence", "Theresa", "Dylan", "Madison", "Jesse", "Beverly",
    "Jordan", "Denise", "Bryan", "Marilyn", "Billy", "Amber", "Joe", "Danielle",
    "Bruce", "Rose", "Gabriel", "Brittany", "Logan", "Diana", "Alan", "Abigail",
    "Juan", "Jane", "Wayne", "Lori", "Roy", "Kathryn", "Ralph", "Alexis",
    "Randy", "Marie", "Eugene", "Olivia", "Vincent", "Frances", "Russell", "Christine",
    "Louis", "Samantha", "Philip", "Jean", "Bobby", "Alice", "Johnny", "Jacqueline"
]

# Common last names
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas", "Taylor",
    "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris", "Sanchez",
    "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams",
    "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
    "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker", "Cruz", "Edwards",
    "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy", "Cook", "Rogers",
    "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey", "Reed", "Kelly",
    "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson", "Watson", "Brooks",
    "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes",
    "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers", "Long", "Ross",
    "Foster", "Jimenez", "Powell", "Jenkins", "Perry", "Russell", "Sullivan", "Bell",
    "Coleman", "Butler", "Henderson", "Barnes", "Gonzales", "Fisher", "Vasquez", "Simmons",
    "Romero", "Jordan", "Patterson", "Alexander", "Hamilton", "Graham", "Reynolds", "Griffin",
    "Wallace", "Moreno", "West", "Cole", "Hayes", "Bryant", "Herrera", "Gibson",
    "Ellis", "Tran", "Medina", "Aguilar", "Stevens", "Murray", "Ford", "Castro",
    "Marshall", "Owens", "Harrison", "Fernandez", "Mcdonald", "Woods", "Washington", "Kennedy"
]

# Email domains (professional)
EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "protonmail.com", "aol.com", "mail.com", "zoho.com", "yandex.com"
]


def generate_candidate_name() -> str:
    """
    Generate a random realistic candidate name.
    
    Returns:
        Full name (first + last)
    """
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    return f"{first_name} {last_name}"


def generate_candidate_email(name: Optional[str] = None, candidate_id: Optional[str] = None) -> str:
    """
    Generate a realistic email address for a candidate.
    
    Args:
        name: Candidate name (if provided, uses it for email)
        candidate_id: Candidate ID (if name not provided, uses ID)
        
    Returns:
        Email address
    """
    domain = random.choice(EMAIL_DOMAINS)
    
    if name:
        # Use name for email: firstname.lastname@domain.com
        name_parts = name.lower().split()
        if len(name_parts) >= 2:
            first = name_parts[0]
            last = name_parts[-1]
            # Add random number to avoid duplicates
            num = random.randint(1, 999)
            return f"{first}.{last}{num}@{domain}"
        else:
            # Fallback if name is single word
            base = name_parts[0].lower()
            num = random.randint(100, 999)
            return f"{base}{num}@{domain}"
    elif candidate_id:
        # Use candidate ID for email
        clean_id = candidate_id.replace("resume", "").replace("_", "").lower()
        num = random.randint(10, 99)
        return f"candidate.{clean_id}{num}@{domain}"
    else:
        # Generate random email
        first = random.choice(FIRST_NAMES).lower()
        last = random.choice(LAST_NAMES).lower()
        num = random.randint(100, 999)
        return f"{first}.{last}{num}@{domain}"


def generate_candidate_info(candidate_id: Optional[str] = None) -> Dict[str, str]:
    """
    Generate candidate name and email.
    
    Args:
        candidate_id: Optional candidate ID for consistent generation
        
    Returns:
        Dictionary with 'name' and 'email'
    """
    name = generate_candidate_name()
    email = generate_candidate_email(name=name, candidate_id=candidate_id)
    
    return {
        "name": name,
        "email": email
    }


def generate_candidate_info_deterministic(candidate_id: str, seed: Optional[int] = None) -> Dict[str, str]:
    """
    Generate deterministic candidate name and email based on candidate ID.
    This ensures the same candidate always gets the same name/email.
    
    Args:
        candidate_id: Candidate ID (e.g., "resume1_123")
        seed: Optional seed for reproducibility
        
    Returns:
        Dictionary with 'name' and 'email'
    """
    # Use candidate ID as seed for deterministic generation
    if seed is None:
        # Create seed from candidate ID
        seed = hash(candidate_id) % (2**31)
    
    random.seed(seed)
    
    # Generate name
    first_idx = seed % len(FIRST_NAMES)
    last_idx = (seed * 7) % len(LAST_NAMES)  # Use different multiplier for variety
    name = f"{FIRST_NAMES[first_idx]} {LAST_NAMES[last_idx]}"
    
    # Generate email based on name
    name_parts = name.lower().split()
    first = name_parts[0]
    last = name_parts[-1]
    domain_idx = (seed * 11) % len(EMAIL_DOMAINS)
    domain = EMAIL_DOMAINS[domain_idx]
    num = (seed % 900) + 100  # 100-999
    
    email = f"{first}.{last}{num}@{domain}"
    
    # Reset random seed
    random.seed()
    
    return {
        "name": name,
        "email": email
    }

