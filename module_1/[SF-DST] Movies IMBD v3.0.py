{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['.ipynb_checkpoints', 'data.csv', '[SF-DST] Movies IMBD v3.0.ipynb']\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "from collections import Counter\n",
    "print(os.listdir(r'C:\\Users\\devba\\Documents\\SF\\imbd-sf'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>imdb_id</th>\n",
       "      <th>popularity</th>\n",
       "      <th>budget</th>\n",
       "      <th>revenue</th>\n",
       "      <th>original_title</th>\n",
       "      <th>cast</th>\n",
       "      <th>director</th>\n",
       "      <th>tagline</th>\n",
       "      <th>overview</th>\n",
       "      <th>runtime</th>\n",
       "      <th>genres</th>\n",
       "      <th>production_companies</th>\n",
       "      <th>release_date</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>release_year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>tt0369610</td>\n",
       "      <td>32.985763</td>\n",
       "      <td>150000000</td>\n",
       "      <td>1513528810</td>\n",
       "      <td>Jurassic World</td>\n",
       "      <td>Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...</td>\n",
       "      <td>Colin Trevorrow</td>\n",
       "      <td>The park is open.</td>\n",
       "      <td>Twenty-two years after the events of Jurassic ...</td>\n",
       "      <td>124</td>\n",
       "      <td>Action|Adventure|Science Fiction|Thriller</td>\n",
       "      <td>Universal Studios|Amblin Entertainment|Legenda...</td>\n",
       "      <td>6/9/2015</td>\n",
       "      <td>5562</td>\n",
       "      <td>6.5</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>tt1392190</td>\n",
       "      <td>28.419936</td>\n",
       "      <td>150000000</td>\n",
       "      <td>378436354</td>\n",
       "      <td>Mad Max: Fury Road</td>\n",
       "      <td>Tom Hardy|Charlize Theron|Hugh Keays-Byrne|Nic...</td>\n",
       "      <td>George Miller</td>\n",
       "      <td>What a Lovely Day.</td>\n",
       "      <td>An apocalyptic story set in the furthest reach...</td>\n",
       "      <td>120</td>\n",
       "      <td>Action|Adventure|Science Fiction|Thriller</td>\n",
       "      <td>Village Roadshow Pictures|Kennedy Miller Produ...</td>\n",
       "      <td>5/13/2015</td>\n",
       "      <td>6185</td>\n",
       "      <td>7.1</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>tt2908446</td>\n",
       "      <td>13.112507</td>\n",
       "      <td>110000000</td>\n",
       "      <td>295238201</td>\n",
       "      <td>Insurgent</td>\n",
       "      <td>Shailene Woodley|Theo James|Kate Winslet|Ansel...</td>\n",
       "      <td>Robert Schwentke</td>\n",
       "      <td>One Choice Can Destroy You</td>\n",
       "      <td>Beatrice Prior must confront her inner demons ...</td>\n",
       "      <td>119</td>\n",
       "      <td>Adventure|Science Fiction|Thriller</td>\n",
       "      <td>Summit Entertainment|Mandeville Films|Red Wago...</td>\n",
       "      <td>3/18/2015</td>\n",
       "      <td>2480</td>\n",
       "      <td>6.3</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>tt2488496</td>\n",
       "      <td>11.173104</td>\n",
       "      <td>200000000</td>\n",
       "      <td>2068178225</td>\n",
       "      <td>Star Wars: The Force Awakens</td>\n",
       "      <td>Harrison Ford|Mark Hamill|Carrie Fisher|Adam D...</td>\n",
       "      <td>J.J. Abrams</td>\n",
       "      <td>Every generation has a story.</td>\n",
       "      <td>Thirty years after defeating the Galactic Empi...</td>\n",
       "      <td>136</td>\n",
       "      <td>Action|Adventure|Science Fiction|Fantasy</td>\n",
       "      <td>Lucasfilm|Truenorth Productions|Bad Robot</td>\n",
       "      <td>12/15/2015</td>\n",
       "      <td>5292</td>\n",
       "      <td>7.5</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>tt2820852</td>\n",
       "      <td>9.335014</td>\n",
       "      <td>190000000</td>\n",
       "      <td>1506249360</td>\n",
       "      <td>Furious 7</td>\n",
       "      <td>Vin Diesel|Paul Walker|Jason Statham|Michelle ...</td>\n",
       "      <td>James Wan</td>\n",
       "      <td>Vengeance Hits Home</td>\n",
       "      <td>Deckard Shaw seeks revenge against Dominic Tor...</td>\n",
       "      <td>137</td>\n",
       "      <td>Action|Crime|Thriller</td>\n",
       "      <td>Universal Pictures|Original Film|Media Rights ...</td>\n",
       "      <td>4/1/2015</td>\n",
       "      <td>2947</td>\n",
       "      <td>7.3</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     imdb_id  popularity     budget     revenue                original_title  \\\n",
       "0  tt0369610   32.985763  150000000  1513528810                Jurassic World   \n",
       "1  tt1392190   28.419936  150000000   378436354            Mad Max: Fury Road   \n",
       "2  tt2908446   13.112507  110000000   295238201                     Insurgent   \n",
       "3  tt2488496   11.173104  200000000  2068178225  Star Wars: The Force Awakens   \n",
       "4  tt2820852    9.335014  190000000  1506249360                     Furious 7   \n",
       "\n",
       "                                                cast          director  \\\n",
       "0  Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...   Colin Trevorrow   \n",
       "1  Tom Hardy|Charlize Theron|Hugh Keays-Byrne|Nic...     George Miller   \n",
       "2  Shailene Woodley|Theo James|Kate Winslet|Ansel...  Robert Schwentke   \n",
       "3  Harrison Ford|Mark Hamill|Carrie Fisher|Adam D...       J.J. Abrams   \n",
       "4  Vin Diesel|Paul Walker|Jason Statham|Michelle ...         James Wan   \n",
       "\n",
       "                         tagline  \\\n",
       "0              The park is open.   \n",
       "1             What a Lovely Day.   \n",
       "2     One Choice Can Destroy You   \n",
       "3  Every generation has a story.   \n",
       "4            Vengeance Hits Home   \n",
       "\n",
       "                                            overview  runtime  \\\n",
       "0  Twenty-two years after the events of Jurassic ...      124   \n",
       "1  An apocalyptic story set in the furthest reach...      120   \n",
       "2  Beatrice Prior must confront her inner demons ...      119   \n",
       "3  Thirty years after defeating the Galactic Empi...      136   \n",
       "4  Deckard Shaw seeks revenge against Dominic Tor...      137   \n",
       "\n",
       "                                      genres  \\\n",
       "0  Action|Adventure|Science Fiction|Thriller   \n",
       "1  Action|Adventure|Science Fiction|Thriller   \n",
       "2         Adventure|Science Fiction|Thriller   \n",
       "3   Action|Adventure|Science Fiction|Fantasy   \n",
       "4                      Action|Crime|Thriller   \n",
       "\n",
       "                                production_companies release_date  vote_count  \\\n",
       "0  Universal Studios|Amblin Entertainment|Legenda...     6/9/2015        5562   \n",
       "1  Village Roadshow Pictures|Kennedy Miller Produ...    5/13/2015        6185   \n",
       "2  Summit Entertainment|Mandeville Films|Red Wago...    3/18/2015        2480   \n",
       "3          Lucasfilm|Truenorth Productions|Bad Robot   12/15/2015        5292   \n",
       "4  Universal Pictures|Original Film|Media Rights ...     4/1/2015        2947   \n",
       "\n",
       "   vote_average  release_year  \n",
       "0           6.5          2015  \n",
       "1           7.1          2015  \n",
       "2           6.3          2015  \n",
       "3           7.5          2015  \n",
       "4           7.3          2015  "
      ]
     },
     "execution_count": 142,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(r'C:\\Users\\devba\\Documents\\SF\\imbd-sf\\data.csv')\n",
    "data.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1890"
      ]
     },
     "execution_count": 143,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Предобработка датасета"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls = [] # создадим список с ответами. сюда будем добавлять ответы по мере прохождения теста\n",
    "# сюда можем вписать создание новых колонок в датасете"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. У какого фильма из списка самый большой бюджет?\n",
    "Варианты ответов:\n",
    "1. The Dark Knight Rises (tt1345836)\n",
    "2. Spider-Man 3 (tt0413300)\n",
    "3. Avengers: Age of Ultron (tt2395427)\n",
    "4. The Warrior's Way\t(tt1032751)\n",
    "5. Pirates of the Caribbean: On Stranger Tides (tt1298650)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [],
   "source": [
    "# тут вводим ваш ответ и добавлем в его список ответов (сейчас для примера стоит \"1\")\n",
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Какой из фильмов самый длительный (в минутах)\n",
    "1. The Lord of the Rings: The Return of the King\t(tt0167260)\n",
    "2. Gods and Generals\t(tt0279111)\n",
    "3. King Kong\t(tt0360717)\n",
    "4. Pearl Harbor\t(tt0213149)\n",
    "5. Alexander\t(tt0346491)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Какой из фильмов самый короткий (в минутах)\n",
    "Варианты ответов:\n",
    "\n",
    "1. Home on the Range\ttt0299172\n",
    "2. The Jungle Book 2\ttt0283426\n",
    "3. Winnie the Pooh\ttt1449283\n",
    "4. Corpse Bride\ttt0121164\n",
    "5. Hoodwinked!\ttt0443536"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Средняя длительность фильма?\n",
    "\n",
    "Варианты ответов:\n",
    "1. 115\n",
    "2. 110\n",
    "3. 105\n",
    "4. 120\n",
    "5. 100\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Средняя длительность фильма по медиане?\n",
    "Варианты ответов:\n",
    "1. 106\n",
    "2. 112\n",
    "3. 101\n",
    "4. 120\n",
    "5. 115\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 6. Какой самый прибыльный фильм?\n",
    "Варианты ответов:\n",
    "1. The Avengers\ttt0848228\n",
    "2. Minions\ttt2293640\n",
    "3. Star Wars: The Force Awakens\ttt2488496\n",
    "4. Furious 7\ttt2820852\n",
    "5. Avatar\ttt0499549"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 7. Какой фильм самый убыточный?\n",
    "Варианты ответов:\n",
    "1. Supernova tt0134983\n",
    "2. The Warrior's Way tt1032751\n",
    "3. Flushed Away\ttt0424095\n",
    "4. The Adventures of Pluto Nash\ttt0180052\n",
    "5. The Lone Ranger\ttt1210819"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 8. Сколько всего фильмов в прибыли?\n",
    "Варианты ответов:\n",
    "1. 1478\n",
    "2. 1520\n",
    "3. 1241\n",
    "4. 1135\n",
    "5. 1398\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 9. Самый прибыльный фильм в 2008 году?\n",
    "Варианты ответов:\n",
    "1. Madagascar: Escape 2 Africa\ttt0479952\n",
    "2. Iron Man\ttt0371746\n",
    "3. Kung Fu Panda\ttt0441773\n",
    "4. The Dark Knight\ttt0468569\n",
    "5. Mamma Mia!\ttt0795421"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 10. Самый убыточный фильм за период с 2012 по 2014 (включительно)?\n",
    "Варианты ответов:\n",
    "1. Winter's Tale\ttt1837709\n",
    "2. Stolen\ttt1656186\n",
    "3. Broken City\ttt1235522\n",
    "4. Upside Down\ttt1374992\n",
    "5. The Lone Ranger\ttt1210819\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 11. Какого жанра фильмов больше всего?\n",
    "Варианты ответов:\n",
    "1. Action\n",
    "2. Adventure\n",
    "3. Drama\n",
    "4. Comedy\n",
    "5. Thriller"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 12. Какого жанра среди прибыльных фильмов больше всего?\n",
    "Варианты ответов:\n",
    "1. Drama\n",
    "2. Comedy\n",
    "3. Action\n",
    "4. Thriller\n",
    "5. Adventure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 13. Кто из режиссеров снял больше всего фильмов?\n",
    "Варианты ответов:\n",
    "1. Steven Spielberg\n",
    "2. Ridley Scott \n",
    "3. Steven Soderbergh\n",
    "4. Christopher Nolan\n",
    "5. Clint Eastwood"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 14. Кто из режиссеров снял больше всего Прибыльных фильмов?\n",
    "Варианты ответов:\n",
    "1. Steven Soderbergh\n",
    "2. Clint Eastwood\n",
    "3. Steven Spielberg\n",
    "4. Ridley Scott\n",
    "5. Christopher Nolan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 15. Кто из режиссеров принес больше всего прибыли?\n",
    "Варианты ответов:\n",
    "1. Steven Spielberg\n",
    "2. Christopher Nolan\n",
    "3. David Yates\n",
    "4. James Cameron\n",
    "5. Peter Jackson\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 16. Какой актер принес больше всего прибыли?\n",
    "Варианты ответов:\n",
    "1. Emma Watson\n",
    "2. Johnny Depp\n",
    "3. Michelle Rodriguez\n",
    "4. Orlando Bloom\n",
    "5. Rupert Grint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 17. Какой актер принес меньше всего прибыли в 2012 году?\n",
    "Варианты ответов:\n",
    "1. Nicolas Cage\n",
    "2. Danny Huston\n",
    "3. Kirsten Dunst\n",
    "4. Jim Sturgess\n",
    "5. Sami Gayle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 18. Какой актер снялся в большем количестве высокобюджетных фильмов? (в фильмах где бюджет выше среднего по данной выборке)\n",
    "Варианты ответов:\n",
    "1. Tom Cruise\n",
    "2. Mark Wahlberg \n",
    "3. Matt Damon\n",
    "4. Angelina Jolie\n",
    "5. Adam Sandler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 19. В фильмах какого жанра больше всего снимался Nicolas Cage?  \n",
    "Варианты ответа:\n",
    "1. Drama\n",
    "2. Action\n",
    "3. Thriller\n",
    "4. Adventure\n",
    "5. Crime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 20. Какая студия сняла больше всего фильмов?\n",
    "Варианты ответа:\n",
    "1. Universal Pictures (Universal)\n",
    "2. Paramount Pictures\n",
    "3. Columbia Pictures\n",
    "4. Warner Bros\n",
    "5. Twentieth Century Fox Film Corporation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 21. Какая студия сняла больше всего фильмов в 2015 году?\n",
    "Варианты ответа:\n",
    "1. Universal Pictures\n",
    "2. Paramount Pictures\n",
    "3. Columbia Pictures\n",
    "4. Warner Bros\n",
    "5. Twentieth Century Fox Film Corporation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 22. Какая студия заработала больше всего денег в жанре комедий за все время?\n",
    "Варианты ответа:\n",
    "1. Warner Bros\n",
    "2. Universal Pictures (Universal)\n",
    "3. Columbia Pictures\n",
    "4. Paramount Pictures\n",
    "5. Walt Disney"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 23. Какая студия заработала больше всего денег в 2012 году?\n",
    "Варианты ответа:\n",
    "1. Universal Pictures (Universal)\n",
    "2. Warner Bros\n",
    "3. Columbia Pictures\n",
    "4. Paramount Pictures\n",
    "5. Lucasfilm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 24. Самый убыточный фильм от Paramount Pictures\n",
    "Варианты ответа:\n",
    "\n",
    "1. K-19: The Widowmaker tt0267626\n",
    "2. Next tt0435705\n",
    "3. Twisted tt0315297\n",
    "4. The Love Guru tt0811138\n",
    "5. The Fighter tt0964517"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 25. Какой Самый прибыльный год (заработали больше всего)?\n",
    "Варианты ответа:\n",
    "1. 2014\n",
    "2. 2008\n",
    "3. 2012\n",
    "4. 2002\n",
    "5. 2015"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 26. Какой Самый прибыльный год для студии Warner Bros?\n",
    "Варианты ответа:\n",
    "1. 2014\n",
    "2. 2008\n",
    "3. 2012\n",
    "4. 2010\n",
    "5. 2015"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 27. В каком месяце за все годы суммарно вышло больше всего фильмов?\n",
    "Варианты ответа:\n",
    "1. Январь\n",
    "2. Июнь\n",
    "3. Декабрь\n",
    "4. Сентябрь\n",
    "5. Май"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 28. Сколько суммарно вышло фильмов летом? (за июнь, июль, август)\n",
    "Варианты ответа:\n",
    "1. 345\n",
    "2. 450\n",
    "3. 478\n",
    "4. 523\n",
    "5. 381"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 29. Какой режисер выпускает (суммарно по годам) больше всего фильмов зимой?\n",
    "Варианты ответов:\n",
    "1. Steven Soderbergh\n",
    "2. Christopher Nolan\n",
    "3. Clint Eastwood\n",
    "4. Ridley Scott\n",
    "5. Peter Jackson"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 30. Какой месяц чаще всего по годам самый прибыльный?\n",
    "Варианты ответа:\n",
    "1. Январь\n",
    "2. Июнь\n",
    "3. Декабрь\n",
    "4. Сентябрь\n",
    "5. Май"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 31. Названия фильмов какой студии в среднем самые длинные по количеству символов?\n",
    "Варианты ответа:\n",
    "1. Universal Pictures (Universal)\n",
    "2. Warner Bros\n",
    "3. Jim Henson Company, The\n",
    "4. Paramount Pictures\n",
    "5. Four By Two Productions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 32. Названия фильмов какой студии в среднем самые длинные по количеству слов?\n",
    "Варианты ответа:\n",
    "1. Universal Pictures (Universal)\n",
    "2. Warner Bros\n",
    "3. Jim Henson Company, The\n",
    "4. Paramount Pictures\n",
    "5. Four By Two Productions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 33. Сколько разных слов используется в названиях фильмов?(без учета регистра)\n",
    "Варианты ответа:\n",
    "1. 6540\n",
    "2. 1002\n",
    "3. 2461\n",
    "4. 28304\n",
    "5. 3432"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 34. Какие фильмы входят в 1 процент лучших по рейтингу?\n",
    "Варианты ответа:\n",
    "1. Inside Out, Gone Girl, 12 Years a Slave\n",
    "2. BloodRayne, The Adventures of Rocky & Bullwinkle\n",
    "3. The Lord of the Rings: The Return of the King\n",
    "4. 300, Lucky Number Slevin"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 35. Какие актеры чаще всего снимаются в одном фильме вместе\n",
    "Варианты ответа:\n",
    "1. Johnny Depp & Helena Bonham Carter\n",
    "2. Hugh Jackman & Ian McKellen\n",
    "3. Vin Diesel & Paul Walker\n",
    "4. Adam Sandler & Kevin James\n",
    "5. Daniel Radcliffe & Rupert Grint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 36. У какого из режиссеров выше вероятность выпустить фильм в прибыли? (5 баллов)101\n",
    "Варианты ответа:\n",
    "1. Quentin Tarantino\n",
    "2. Steven Soderbergh\n",
    "3. Robert Rodriguez\n",
    "4. Christopher Nolan\n",
    "5. Clint Eastwood"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Submission"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "36"
      ]
     },
     "execution_count": 181,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(answer_ls)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>Answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>11</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>13</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>14</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>15</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>16</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>17</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>18</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>19</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>20</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>21</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>22</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>23</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>24</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>25</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>26</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>27</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>28</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>29</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>30</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>31</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31</th>\n",
       "      <td>32</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>33</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>33</th>\n",
       "      <td>34</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>34</th>\n",
       "      <td>35</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35</th>\n",
       "      <td>36</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Id  Answer\n",
       "0    1       4\n",
       "1    2       2\n",
       "2    3       3\n",
       "3    4       2\n",
       "4    5       1\n",
       "5    6       5\n",
       "6    7       2\n",
       "7    8       1\n",
       "8    9       4\n",
       "9   10       5\n",
       "10  11       3\n",
       "11  12       1\n",
       "12  13       3\n",
       "13  14       4\n",
       "14  15       5\n",
       "15  16       1\n",
       "16  17       3\n",
       "17  18       3\n",
       "18  19       2\n",
       "19  20       1\n",
       "20  21       4\n",
       "21  22       2\n",
       "22  23       3\n",
       "23  24       1\n",
       "24  25       5\n",
       "25  26       1\n",
       "26  27       4\n",
       "27  28       2\n",
       "28  29       3\n",
       "29  30       2\n",
       "30  31       5\n",
       "31  32       5\n",
       "32  33       3\n",
       "33  34       3\n",
       "34  35       5\n",
       "35  36       4"
      ]
     },
     "execution_count": 182,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.DataFrame({'Id':range(1,len(answer_ls)+1), 'Answer':answer_ls}, columns=['Id', 'Answer'])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
