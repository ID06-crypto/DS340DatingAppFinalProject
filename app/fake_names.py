"""Display names. Every participant is shown to others under a name picked from these
lists, by gender, so nobody sees anyone's real name. Nonbinary participants get a name
from both lists. The real name is kept privately, in the private_names table.
"""

import random

MEN = [
    "James", "Michael", "John", "Robert", "David", "William", "Richard", "Joseph",
    "Thomas", "Christopher", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
    "Steven", "Andrew", "Donald", "Joshua", "Paul", "Kevin", "Kenneth", "Brian",
    "Timothy", "Ronald", "Jason", "George", "Edward", "Jeffrey", "Jacob", "Ryan",
    "Nicholas", "Gary", "Eric", "Jonathan", "Stephen", "Larry", "Justin",
    "Benjamin", "Scott", "Brandon", "Samuel", "Alexander", "Gregory", "Patrick",
    "Jack", "Frank", "Raymond", "Dennis", "Aaron", "Tyler", "Jerry", "Jose",
    "Nathan", "Adam", "Henry", "Zachary", "Douglas", "Peter", "Noah", "Ethan",
    "Kyle", "Christian", "Jeremy", "Austin", "Keith", "Sean", "Terry", "Roger",
    "Dylan", "Walter", "Gerald", "Jordan", "Gabriel", "Carl", "Bryan", "Jesse",
    "Logan", "Lawrence", "Elijah", "Arthur", "Bruce", "Harold", "Billy", "Liam",
    "Alan", "Juan", "Joe", "Mason", "Lucas", "Randy", "Willie", "Wayne",
    "Vincent", "Caleb", "Albert", "Luke", "Isaac", "Bradley", "Cameron",
]

WOMEN = [
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
    "Jessica", "Karen", "Sarah", "Lisa", "Nancy", "Sandra", "Ashley", "Emily",
    "Kimberly", "Michelle", "Donna", "Margaret", "Carol", "Betty", "Amanda",
    "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Cynthia", "Laura",
    "Amy", "Kathleen", "Angela", "Emma", "Shirley", "Dorothy", "Brenda", "Nicole",
    "Pamela", "Samantha", "Anna", "Olivia", "Katherine", "Christine", "Debra",
    "Rachel", "Maria", "Carolyn", "Janet", "Heather", "Diane", "Julie",
    "Catherine", "Victoria", "Joyce", "Lauren", "Kelly", "Christina", "Helen",
    "Joan", "Hannah", "Judith", "Andrea", "Evelyn", "Megan", "Cheryl", "Ruth",
    "Virginia", "Sophia", "Madison", "Jacqueline", "Abigail", "Isabella",
    "Teresa", "Sara", "Charlotte", "Janice", "Kathryn", "Martha", "Gloria", "Ann",
    "Judy", "Amber", "Danielle", "Denise", "Grace", "Julia", "Natalie", "Diana",
    "Marilyn", "Brittany", "Beverly", "Alice", "Theresa", "Jean", "Kayla",
    "Alexis", "Frances", "Ava", "Tiffany", "Lori",
]

NAMES_BY_GENDER = {"man": MEN, "woman": WOMEN, "nonbinary": MEN + WOMEN}


def pick_name(gender: str, in_use: set[str], rng: random.Random) -> str:
    """A random name for this gender that nobody in the study has yet, if one is left.

    With about 100 names per list, repeats only happen once a list runs out.
    """
    names = NAMES_BY_GENDER.get(gender, MEN + WOMEN)
    free = [name for name in names if name not in in_use]
    return rng.choice(free or names)
