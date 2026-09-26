import copy
import unittest
from .refresh_gs64 import replace_rows


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(candidate_id=str(i),architecture={'q8_group_size':g},split='train',
                        permutation_group=str(i),metrics={'tp':i},cohort='original') for i,g in enumerate((16,32,64))]
        self.new={'2':dict(self.rows[2],metrics={'tp':99})}

    def test_identity_order_and_old_rows_unchanged(self):
        original=copy.deepcopy(self.rows)
        result=replace_rows(self.rows,self.new)
        self.assertEqual(self.rows,original)
        self.assertEqual(result[:2],original[:2])
        self.assertEqual(result[2]['metrics'],{'tp':99})
        self.assertEqual([r['candidate_id'] for r in result],['0','1','2'])

    def test_incomplete_subset_rejected(self):
        with self.assertRaisesRegex(ValueError,'complete GS64'):replace_rows(self.rows,{})

    def test_extra_id_rejected(self):
        with self.assertRaisesRegex(ValueError,'complete GS64'):replace_rows(self.rows,dict(self.new,extra=self.rows[0]))

    def test_changed_split_rejected(self):
        self.new['2']=dict(self.new['2'],split='test')
        with self.assertRaisesRegex(ValueError,'split'):replace_rows(self.rows,self.new)

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):replace_rows(self.rows+self.rows[:1],self.new)


if __name__=='__main__':unittest.main()
