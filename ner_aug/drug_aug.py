import json
import random

class DrugAug():
	def __init__(self):
		self.drug_map = {}

		DRUG_PATH = "/Users/annisaningtyas/Documents/GitHub/data-augmentation/resource/drugs_dict.json"
		self.__build_drug_map(DRUG_PATH)

	def __map_drug(self, key, value):
		key = key.lower()
		val_clean = [v for v in value if v.count('-')<3]

		if key not in self.drug_map:
			# heuristic to skip if the synonym contain chemical characters
			self.drug_map[key] = val_clean
		else:
			self.drug_map[key].extend(val_clean)
			# self.drug_map[key] = self.drug_map[key] + list(set(add_list) - set(self.drug_map[key]))

	def __build_drug_map(self, filepath):
		with open(filepath) as json_file:
			data = json.load(json_file)
			for d in data:
				self.__map_drug(d['drug_name'], d['synonyms'])

				# self.__map_drug(d['drug_name'], d['synonyms'], d['products_name'])

				# for p_name in d['products_name']:
				#	self.__map_drug(p_name, d['synonyms'], [d['drug_name']])

				# for s_name in d['synonyms']:
				#	self.__map_drug(s_name, d['products_name'], [d['drug_name']])

	def get_synonyms(self, key):
		if key in self.drug_map:
			return self.drug_map[key]
		else:
			return []

	def augment(self, text):
		augmented_text = []
		tokens = text.split()
		for token in tokens:
			syn = self.get_synonyms(token)
			if len(syn) > 0:
				augmented_text.append(random.choice(syn))
			else:
				augmented_text.append(token)

		augmented_text = ' '.join(augmented_text)

		if len(augmented_text.strip()) == 0:
			return text
		else:
			return augmented_text