from preprocess import *
from database import *
import pandas as pd

def main():
    preprocess = Preprocess()
    database = Database('text')


    data = pd.DataFrame({'number': [6, 5, 3, 69, 420], 'text': ["WEBVTT 00:00:00.320 --> 00:00:01.620 You don't even have a cat. 00:00:01.760 --> 00:00:02.820 But for cat, 00:00:03.000 --> 00:00:03.820 what's it for",
                                                       "WEBVTT 00:00:00.088 --> 00:00:02.652 taking your grandparents to Duckcon 1",
                                                       "WEBVTT 00:00:00.920 --> 00:00:04.460 wat betekenen de letters voc verzopen op camping",
                                                       "WEBVTT 00:00:00.920 --> 00:00:04.460 Geen idee wat ik hier nou moet schrijven. Weet je zeker dat dit goed gaat? Ik wil even lekker dansen. vandaag gaan we geïmproviseerd mosselgerechtje maken want fleur heeft geen zin in dunya als het niet hoeft liever geen dunya zei je gelukkig ben ik zo creatief dat ik gewoon iets anders maken en dat lekker basic maar wel echt een supergoed recept al zeg ik het zelf snijd ik knoflook citroen en witte ui dan gaat er roomboter en ansjovis in de grote pan kijk ik had in één keer een nieuwe hoek gevonden veel beter oké die ui en die knoflook die citroen gaan erin dan twee blaadjes laurier gaan erbij dan gaat er flink wat mosterd bij en dat blussen we af met witte wijn vervolgens ook de kookroom beetje romig wordt dan mogen de mosselen erbij dan gaan we het even mix ja even zo mengen door elkaar tot alle mosselen open staan en dan is dit het eindresultaat en ik maakte deze super lekkere pasta erbij. de eredivisie gaat eindelijk weer van start en dit zijn mijn voorspellingen voor het linker rijtje op 9 ga ik voor sparta na een goed seizoen vorig jaar hebben ze een groot deel van de selectie weten te houden en negende lijn me dan ook prima dan vitesse na een dramatisch jaar vorig seizoen hoop ik dat het dit jaar wat rustiger wordt in arnhem het achtste is dan ook goed nummer 7 wordt heerenveen weer een leuke selectie dit jaar en ze strijden ieder jaar waarmee een play offs voor europees voetbal optisch ga ik voor twente na een geweldig jaar zijn ze rond jans en een aantal belangrijke spelers verloren sinds dit lijkt me nog altijd prima 1 op 5 ga ik voor utrecht de club heeft heel veel nieuwe leuke spelers gehaald en ik ben heel benieuwd hoe ze gaan presteren dan de top 4 om te beginnen met az zij zijn een aantal hele belangrijke spelers verloren maar hebben nog altijd een hele leuke selectie en kunnen daarmee denk ik nog altijd vierde woorden de top 3 begint met psv ondanks dat ik hoop dat psv mee kan spelen om de titel denk ik dat ze onnodige punten gaan verliezen en er moeten nog een aantal nieuwe krachten bij de selectie komen net geen kampioen gaat ajax worden het rommel een beetje binnen de club en het is maar de vraag hoe muri stijn en de nieuwe aankoop het gaan doen en opnieuw kampioen feyenoord ze hebben een groot deel van de selectie bij elkaar weten te houden en met slot aan het roer denk ik dat ze de meeste kans maken wat voorspel jij voor komend seizoen ", '{"utterances": [{"text": "do you recognize this song", "start_time": 0, "end_time": 1372, "words": null, "text_size": 28, "text_color": "#FFFFFFFF", "bg_color": "#00000000", "alignment": 0, "source_width": 0.8647342995169082, "source_height": 0.07674449812130972}, {"text": "this song is Harness Your Hopes by the band pavement", "start_time": 4733, "end_time": 7686, "words": null, "text_size": 28, "text_color": "#FFFFFFFF", "bg_color": "#00000000", "alignment": 0, "source_width": 0.8176328502415459, "source_height": 0.16560654857756307}, {"text": "released in 1999", "start_time": 7693, "end_time": 9606, "words": null, "text_size": 28, "text_color": "#FFFFFFFF", "bg_color": "#00000000", "alignment": 0, "source_width": 0.5507246376811594, "source_height": 0.07674449812130972}, {"text": "it is currently the bands most streamed song on Spotify", "start_time": 9720, "end_time": 12546, "words": null, "text_size": 28}]}'], 'language_id': [2, 2, 18, 18, 2]})
    print(data)

    database.insert_data(
        tableName='test',
        data=data
    )

    df = database.retrieve_data(
        tableName='test',
        colNames=['id', 'number', 'text', 'language_id'],
        batch_size=5
    )

    cleaned = preprocess.preprocess_text(df, 'text')
    translated = preprocess.generate_translation(cleaned, 'text', 'language_id')
    filtered = preprocess.filter_on_heuristics(translated, 'translation')

    database.update_data(
        tableName='test',
        data=filtered
    )


if __name__ == '__main__':
    main()