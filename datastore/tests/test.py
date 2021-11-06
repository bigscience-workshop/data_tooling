import os
import sys

import aiohttp
import fsspec
import numpy as np
import requests
from datasets import load_dataset

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir
        )
    )
)

from data_tooling.datastore.datastore_base import *

if __name__ == "__main__":

    args = sys.argv[1:]
    if not args:
        exit()

    if "-test_sql_basic" in args:
        db = DatabaseExt("sqlite://")
        table = db["user"]
        assert table.exists == False
        assert table.has_column("id") == False
        assert table.insert(dict(name="John Doe", age=37)) == 0
        assert table.exists == True
        assert table.has_column("id") == True
        assert table.insert(dict(name="Jane Doe", age=20)) == 1
        jane = table.find_one(name="Jane Doe")
        assert jane["id"] == 1

        db = DatabaseExt("sqlite://")
        table = db["user"]
        rows = [dict(name="Dolly")] * 10
        table.insert_many(rows)
        assert list(table.find(id={"in": range(0, 2)}))[-1]["id"] == 1

    if "-test_memmap" in args:
        datastore = Datastore.from_mmap(
            "embed",
            [1000, 512, 512],
        )
        print(datastore)
        datastore = Datastore.from_dataset(
            load_dataset("oscar", "unshuffled_deduplicated_sw")["train"]
        )
        datastore = datastore.add_mmap(
            "embed",
            [-1, 512, 512],
        )
        datastore = datastore.add_mmap("token", [-1, 512], dtype=np.int32)
        assert (datastore["embed"][0].shape) == (512, 512)
        datastore["embed"][0][0] = 0.0
        assert np.mean(datastore["embed"][0][0]) == 0
        datastore["embed"][0][0] = 1.0
        assert np.mean(datastore["embed"][0][0]) == 1.0
        assert set(datastore[0].keys()) == {"id", "text", "embed", "token"}
        assert len(datastore["text"]) == 24803
        assert len(datastore[0:10]["text"]) == 10
        assert (datastore[0:10]["token"].shape) == (10, 512)
        print(datastore)

    if "-test_move_to_sql" in args:
        data = load_dataset("oscar", "unshuffled_deduplicated_yo")["train"]
        datastore = Datastore.from_dataset(data)
        Datastore.db_connection = {}
        Datastore.db_table = {}
        datastore = datastore.move_to_sql("text")

    if "-test_sqlite_fs" in args:
        db = DatabaseExt("sqlite:///test.db")
        books_table = db["books"]
        books_table.create_fts_index_column("text", stemmer="unicode61")
        for t in """In Which We Are Introduced to Winnie the Pooh and Some Bees and the Stories Begin
Winnie-the-Pooh is out of honey, so he and Christopher Robin attempt to trick some bees out of theirs, with disastrous results.
In Which Pooh Goes Visiting and Gets into a Tight Place
Pooh visits Rabbit, but eats so much while in Rabbit's house that he gets stuck in Rabbit's door on the way out.
In Which Pooh and Piglet Go Hunting and Nearly Catch a Woozle
Pooh and Piglet track increasing numbers of footsteps round and round a spinney of trees.
In Which Eeyore Loses a Tail and Pooh Finds One
Pooh sets out to find Eeyore's missing tail, and notices something interesting about Owl's bell-pull.
In Which Piglet Meets a Heffalump
Piglet and Pooh try to trap a Heffalump, but wind up trapping the wrong sort of creature.
In Which Eeyore has a Birthday and Gets Two Presents
Pooh feels bad that no one has gotten Eeyore anything for his birthday, so he and Piglet try their best to get him presents.
In Which Kanga and Baby Roo Come to the Forest and Piglet has a Bath
Rabbit convinces Pooh and Piglet to try to kidnap newcomer Baby Roo to convince newcomer Kanga to leave the forest.
In Which Christopher Robin Leads an Expotition to the North Pole
Christopher Robin and all of the animals in the forest go on a quest to find the North Pole in the Hundred Acre Wood.
In Which Piglet is Entirely Surrounded by Water
Piglet is trapped in his home by a flood, so he sends a message out in a bottle in hope of rescue.
In Which Christopher Robin Gives Pooh a Party and We Say Goodbye
Christopher Robin gives Pooh a party for helping to rescue Piglet during the flood.
The central character in the series is Harry Potter, a boy who lives in the fictional town of Little Whinging, Surrey with his aunt, uncle, and cousin – the Dursleys – and discovers at the age of eleven that he is a wizard, though he lives in the ordinary world of non-magical people known as Muggles.[8] The wizarding world exists parallel to the Muggle world, albeit hidden and in secrecy. His magical ability is inborn, and children with such abilities are invited to attend exclusive magic schools that teach the necessary skills to succeed in the wizarding world.[9]

Harry becomes a student at Hogwarts School of Witchcraft and Wizardry, a wizarding academy in Scotland, and it is here where most of the events in the series take place. As Harry develops through his adolescence, he learns to overcome the problems that face him: magical, social, and emotional, including ordinary teenage challenges such as friendships, infatuation, romantic relationships, schoolwork and exams, anxiety, depression, stress, and the greater test of preparing himself for the confrontation that lies ahead in wizarding Britain's increasingly-violent second wizarding war.[10]

Each novel chronicles one year in Harry's life[11] during the period from 1991 to 1998.[12] The books also contain many flashbacks, which are frequently experienced by Harry viewing the memories of other characters in a device called a Pensieve.

The environment Rowling created is intimately connected to reality. The British magical community of the Harry Potter books is inspired by 1990s British culture, European folklore, classical mythology and alchemy, incorporating objects and wildlife such as magic wands, magic plants, potions, spells, flying broomsticks, centaurs and other magical creatures, and the Philosopher's Stone, beside others invented by Rowling. While the fantasy land of Narnia is an alternate universe and the Lord of the Rings' Middle-earth a mythic past, the wizarding world of Harry Potter exists parallel to the real world and contains magical versions of the ordinary elements of everyday life, with the action mostly set in Scotland (Hogwarts), the West Country, Devon, London, and Surrey in southeast England.[13] The world only accessible to wizards and magical beings comprises a fragmented collection of overlooked hidden streets, ancient pubs, lonely country manors, and secluded castles invisible to the Muggle population.[9]

Early years
When the first novel of the series, Harry Potter and the Philosopher's Stone, opens, it is apparent that some significant event has taken place in the wizarding world – an event so very remarkable that even Muggles (non-magical people) notice signs of it. The full background to this event and Harry Potter's past is revealed gradually throughout the series. After the introductory chapter, the book leaps forward to a time shortly before Harry Potter's eleventh birthday, and it is at this point that his magical background begins to be revealed.

Despite Harry's aunt and uncle's desperate prevention of Harry learning about his abilities,[14] their efforts are in vain. Harry meets a half-giant, Rubeus Hagrid, who is also his first contact with the wizarding world. Hagrid reveals himself to be the Keeper of Keys and Grounds at Hogwarts as well as some of Harry's history.[14] Harry learns that, as a baby, he witnessed his parents' murder by the power-obsessed dark wizard Lord Voldemort (more commonly known by the magical community as You-Know-Who or He-Who-Must-Not-Be-Named, and by Albus Dumbledore as Tom Marvolo Riddle) who subsequently attempted to kill him as well.[14] Instead, the unexpected happened: Harry survived with only a lightning-shaped scar on his forehead as a memento of the attack, and Voldemort disappeared soon afterwards, gravely weakened by his own rebounding curse.

As its inadvertent saviour from Voldemort's reign of terror, Harry has become a living legend in the wizarding world. However, at the orders of the venerable and well-known wizard Albus Dumbledore, the orphaned Harry had been placed in the home of his unpleasant Muggle relatives, the Dursleys, who have kept him safe but treated him poorly, including confining him to a cupboard without meals and treating him as their servant. Hagrid then officially invites Harry to attend Hogwarts School of Witchcraft and Wizardry, a famous magic school in Scotland that educates young teenagers on their magical development for seven years, from age eleven to seventeen.

With Hagrid's help, Harry prepares for and undertakes his first year of study at Hogwarts. As Harry begins to explore the magical world, the reader is introduced to many of the primary locations used throughout the series. Harry meets most of the main characters and gains his two closest friends: Ron Weasley, a fun-loving member of an ancient, large, happy, but poor wizarding family, and Hermione Granger, a gifted, bright, and hardworking witch of non-magical parentage.[14][15] Harry also encounters the school's potions master, Severus Snape, who displays a conspicuously deep and abiding dislike for him, the rich brat Draco Malfoy whom he quickly makes enemies with, and the Defence Against the Dark Arts teacher, Quirinus Quirrell, who later turns out to be allied with Lord Voldemort. He also discovers a talent of flying on broomsticks and is recruited for his house's Quidditch team, a sport in the wizarding world where players fly on broomsticks. The first book concludes with Harry's second confrontation with Lord Voldemort, who, in his quest to regain a body, yearns to gain the power of the Philosopher's Stone, a substance that bestows everlasting life and turns any metal into pure gold.[14]

The series continues with Harry Potter and the Chamber of Secrets, describing Harry's second year at Hogwarts. He and his friends investigate a 50-year-old mystery that appears uncannily related to recent sinister events at the school. Ron's younger sister, Ginny Weasley, enrols in her first year at Hogwarts, and finds an old notebook in her belongings which turns out to be the diary of a previous student, Tom Marvolo Riddle, written during World War II. He is later revealed to be Voldemort's younger self, who is bent on ridding the school of "mudbloods", a derogatory term describing wizards and witches of non-magical parentage. The memory of Tom Riddle resides inside of the diary and when Ginny begins to confide in the diary, Voldemort is able to possess her.

Through the diary, Ginny acts on Voldemort's orders and unconsciously opens the "Chamber of Secrets", unleashing an ancient monster, later revealed to be a basilisk, which begins attacking students at Hogwarts. It kills those who make direct eye contact with it and petrifies those who look at it indirectly. The book also introduces a new Defence Against the Dark Arts teacher, Gilderoy Lockhart, a highly cheerful, self-conceited wizard with a pretentious facade, later turning out to be a fraud. Harry discovers that prejudice exists in the Wizarding World through delving into the school's history, and learns that Voldemort's reign of terror was often directed at wizards and witches who were descended from Muggles.

Harry also learns that his ability to speak the snake language Parseltongue is rare and often associated with the Dark Arts. When Hermione is attacked and petrified, Harry and Ron finally piece together the puzzles and unlock the Chamber of Secrets, with Harry destroying the diary for good and saving Ginny, and, as they learn later, also destroying a part of Voldemort's soul. The end of the book reveals Lucius Malfoy, Draco's father and rival of Ron and Ginny's father, to be the culprit who slipped the book into Ginny's belongings.

The third novel, Harry Potter and the Prisoner of Azkaban, follows Harry in his third year of magical education. It is the only book in the series which does not feature Lord Voldemort in any form, only being mentioned. Instead, Harry must deal with the knowledge that he has been targeted by Sirius Black, his father's best friend, and, according to the Wizarding World, an escaped mass murderer who assisted in the murder of Harry's parents. As Harry struggles with his reaction to the dementors – dark creatures with the power to devour a human soul and feed on despair – which are ostensibly protecting the school, he reaches out to Remus Lupin, a Defence Against the Dark Arts teacher who is eventually revealed to be a werewolf. Lupin teaches Harry defensive measures which are well above the level of magic generally executed by people his age. Harry comes to know that both Lupin and Black were best friends of his father and that Black was framed by their fourth friend, Peter Pettigrew, who had been hiding as Ron's pet rat, Scabbers.[16] In this book, a recurring theme throughout the series is emphasised – in every book there is a new Defence Against the Dark Arts teacher, none of whom lasts more than one school year.

Voldemort returns
"The Elephant House", a small, painted red café where Rowling wrote a few chapters of Harry Potter and the Philosopher's Stone
The Elephant House was one of the cafés in Edinburgh where Rowling wrote the first part of Harry Potter.

The former 1st floor Nicholson's Cafe now renamed Spoon in Edinburgh where J. K. Rowling wrote the first few chapters of Harry Potter and the Philosopher’s Stone.

The J. K. Rowling plaque on the corner of the former Nicholson's Cafe (now renamed Spoon) at 6A Nicolson St, Edinburgh.
During Harry's fourth year of school (detailed in Harry Potter and the Goblet of Fire), Harry is unwillingly entered as a participant in the Triwizard Tournament, a dangerous yet exciting contest where three "champions", one from each participating school, must compete with each other in three tasks in order to win the Triwizard Cup. This year, Harry must compete against a witch and a wizard "champion" from overseas schools Beauxbatons and Durmstrang, as well as another Hogwarts student, causing Harry's friends to distance themselves from him.[17]

Harry is guided through the tournament by their new Defence Against the Dark Arts professor, Alastor "Mad-Eye" Moody, who turns out to be an impostor – one of Voldemort's supporters named Barty Crouch, Jr. in disguise, who secretly entered Harry's name into the tournament. The point at which the mystery is unravelled marks the series' shift from foreboding and uncertainty into open conflict. Voldemort's plan to have Crouch use the tournament to bring Harry to Voldemort succeeds. Although Harry manages to escape, Cedric Diggory, the other Hogwarts champion in the tournament, is killed by Peter Pettigrew and Voldemort re-enters the Wizarding World with a physical body.

In the fifth book, Harry Potter and the Order of the Phoenix, Harry must confront the newly resurfaced Voldemort. In response to Voldemort's reappearance, Dumbledore re-activates the Order of the Phoenix, a secret society which works from Sirius Black's dark family home to defeat Voldemort's minions and protect Voldemort's targets, especially Harry. Despite Harry's description of Voldemort's recent activities, the Ministry of Magic and many others in the magical world refuse to believe that Voldemort has returned. In an attempt to counter and eventually discredit Dumbledore, who along with Harry is the most prominent voice in the Wizarding World attempting to warn of Voldemort's return, the Ministry appoints Dolores Umbridge as the High Inquisitor of Hogwarts and the new Defence Against the Dark Arts teacher. She transforms the school into a dictatorial regime and refuses to allow the students to learn ways to defend themselves against dark magic.[18]

Hermione and Ron form "Dumbledore's Army", a secret study group in which Harry agrees to teach his classmates the higher-level skills of Defence Against the Dark Arts that he has learned from his previous encounters with Dark wizards. Through those lessons, Harry begins to develop a crush on the popular and attractive Cho Chang. Juggling schoolwork, Umbridge's incessant and persistent efforts to land him in trouble and the defensive lessons, Harry begins to lose sleep as he constantly receives disturbing dreams about a dark corridor in the Ministry of Magic, followed by a burning desire to learn more. An important prophecy concerning Harry and Lord Voldemort is then revealed,[19] and Harry discovers that he and Voldemort have a painful connection, allowing Harry to view some of Voldemort's actions telepathically. In the novel's climax, Harry is tricked into seeing Sirius tortured and races to the Ministry of Magic. He and his friends face off against Voldemort's followers (nicknamed Death Eaters) at the Ministry of Magic. Although the timely arrival of members of the Order of the Phoenix saves the teenagers' lives, Sirius Black is killed in the conflict.

In the sixth book, Harry Potter and the Half-Blood Prince, Voldemort begins waging open warfare. Harry and his friends are relatively protected from that danger at Hogwarts. They are subject to all the difficulties of adolescence – Harry eventually begins dating Ginny, Ron establishes a strong infatuation with fellow Hogwarts student Lavender Brown, and Hermione starts to develop romantic feelings towards Ron. Near the beginning of the novel, lacking his own book, Harry is given an old potions textbook filled with many annotations and recommendations signed by a mysterious writer titled; "the Half-Blood Prince". This book is a source of scholastic success and great recognition from their new potions master, Horace Slughorn, but because of the potency of the spells that are written in it, becomes a source of concern.

With war drawing near, Harry takes private lessons with Dumbledore, who shows him various memories concerning the early life of Voldemort in a device called a Pensieve. These reveal that in order to preserve his life, Voldemort has split his soul into pieces, used to create a series of Horcruxes – evil enchanted items hidden in various locations, one of which was the diary destroyed in the second book.[20] Draco, who has joined with the Death Eaters, attempts to attack Dumbledore upon his return from collecting a Horcrux, and the book culminates in the killing of Dumbledore by Professor Snape, the titular Half-Blood Prince.

Harry Potter and the Deathly Hallows, the last original novel in the series, begins directly after the events of the sixth book. Lord Voldemort has completed his ascension to power and gained control of the Ministry of Magic. Harry, Ron and Hermione drop out of school so that they can find and destroy Voldemort's remaining Horcruxes. To ensure their own safety as well as that of their family and friends, they are forced to isolate themselves. A ghoul pretends to be Ron ill with a contagious disease, Harry and the Dursleys separate, and Hermione wipes her parents' memories and sends them abroad.

As the trio searches for the Horcruxes, they learn details about an ancient prophecy of the Deathly Hallows, three legendary items that when united under one Keeper, would supposedly allow that person to be the Master of Death. Harry discovers his handy Invisibility Cloak to be one of those items, and Voldemort to be searching for another: the Elder Wand, the most powerful wand in history. At the end of the book, Harry and his friends learn about Dumbledore's past, as well as Snape's true motives – he had worked on Dumbledore's behalf since the murder of Harry's mother. Eventually, Snape is killed by Voldemort out of paranoia.

The book culminates in the Battle of Hogwarts. Harry, Ron and Hermione, in conjunction with members of the Order of the Phoenix and many of the teachers and students, defend Hogwarts from Voldemort, his Death Eaters, and various dangerous magical creatures. Several major characters are killed in the first wave of the battle, including Remus Lupin and Fred Weasley, Ron's older brother. After learning that he himself is a Horcrux, Harry surrenders himself to Voldemort in the Forbidden Forest, who casts a killing curse (Avada Kedavra) at him. The defenders of Hogwarts do not surrender after learning of Harry's presumed death and continue to fight on. Harry awakens and faces Voldemort, whose Horcruxes have all been destroyed. In the final battle, Voldemort's killing curse rebounds off Harry's defensive spell (Expelliarmus), killing Voldemort.

An epilogue "Nineteen Years Later"[21] describes the lives of the surviving characters and the effects of Voldemort's death on the Wizarding World. In the epilogue, Harry and Ginny are married with three children, and Ron and Hermione are married with two children.[22]
""".split(
            "\n"
        ):
            if t.strip():
                books_table.insert({"text": t})
        print(list(books_table.find(id={"in": (3, 4)})))
        print(list(books_table.find()))
        print("Bottle*", list(books_table.find(_fts_query=[("text", "Bottle*")])))
        print("robin", list(books_table.find(_fts_query=[("text", "robin")])))
        print("Home", list(books_table.find(_fts_query=[("text", "Home")])))
        print("notic*", list(books_table.find(_fts_query=[("text", "notic*")])))
        print(
            "sign* AND notic*",
            list(books_table.find(_fts_query=[("text", "sign* AND notic*")])),
        )
        print(
            "sign AND notic*",
            list(books_table.find(_fts_query=[("text", "sign AND notic*")])),
        )
